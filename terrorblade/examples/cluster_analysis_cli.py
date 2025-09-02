#!/usr/bin/env python3
"""
Cluster Analysis CLI Tool for Terrorblade

This tool provides an interactive CLI for analyzing message clusters with user-friendly output.
Features:
- List all large clusters in a chat (by chat name or ID)
- Analyze specific clusters with detailed statistics
- Show cluster intensity over time
- Summarize clusters using OpenAI GPT models
- Extract stories from cluster discussions
- User-friendly output with names instead of IDs

Usage:
    python cluster_analysis_cli.py --phone +1234567890
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from dotenv import load_dotenv

# Add terrorblade to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from terrorblade.data.database.telegram_database import TelegramDatabase
from terrorblade.utils.config import get_db_path

# Load environment variables
load_dotenv()

# Check for optional dependencies
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ClusterAnalyzer:
    """Main class for cluster analysis functionality."""

    def __init__(self, phone: str, db_path: str = "auto"):
        self.phone = phone
        self.phone_clean = phone.replace("+", "")

        # Resolve database path using centralized config
        self.db_path = get_db_path(db_path)

        self.db = TelegramDatabase(db_path=self.db_path, read_only=True)

        # Table names for this user
        self.messages_table = f"messages_{self.phone_clean}"
        self.clusters_table = f"message_clusters_{self.phone_clean}"
        self.chat_names_table = f"chat_names_{self.phone_clean}"
        self.user_names_table = f"user_names_{self.phone_clean}"

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, "db"):
            self.db.close()

    def _check_tables_exist(self) -> bool:
        """Check if required tables exist for this user."""
        try:
            # Check if the main tables exist by trying to query them
            tables_to_check = [self.messages_table, self.clusters_table, self.chat_names_table, self.user_names_table]

            for table in tables_to_check:
                try:
                    # Try to query the table - if it exists, this will work
                    self.db.db.execute(f"SELECT COUNT(*) FROM {table} LIMIT 1").fetchone()
                except Exception:
                    # If any table doesn't exist or can't be queried, return False
                    return False
            return True
        except Exception:
            return False

    def get_chats_list(self) -> pl.DataFrame:
        """Get list of all chats with names and basic stats."""
        if not self._check_tables_exist():
            print(f"❌ No data found for phone {self.phone}")
            print("💡 Make sure you have:")
            print("   1. Imported your Telegram data")
            print("   2. Used the correct phone number")
            print("   3. Run the data processing pipeline")
            return pl.DataFrame()

        try:
            sql = f"""
            WITH latest_chat_names AS (
                SELECT chat_id, chat_name FROM (
                    SELECT chat_id, chat_name,
                           ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY COALESCE(last_seen, first_seen) DESC) AS rn
                    FROM {self.chat_names_table}
                ) t WHERE rn = 1
            ),
            chat_stats AS (
                SELECT
                    m.chat_id,
                    COUNT(*) as message_count,
                    COUNT(DISTINCT m.from_id) as participant_count,
                    MIN(m.date) as first_message,
                    MAX(m.date) as last_message
                FROM {self.messages_table} m
                GROUP BY m.chat_id
            ),
            cluster_stats AS (
                SELECT
                    c.chat_id,
                    COUNT(DISTINCT c.group_id) as cluster_count,
                    COUNT(*) as clustered_messages,
                    AVG(cluster_size) as avg_cluster_size,
                    MAX(cluster_size) as max_cluster_size
                FROM {self.clusters_table} c
                JOIN (
                    SELECT group_id, chat_id, COUNT(*) as cluster_size
                    FROM {self.clusters_table}
                    GROUP BY group_id, chat_id
                ) sizes ON c.group_id = sizes.group_id AND c.chat_id = sizes.chat_id
                GROUP BY c.chat_id
            )
            SELECT
                cs.chat_id,
                COALESCE(lcn.chat_name, 'Unknown Chat') as chat_name,
                cs.message_count,
                cs.participant_count,
                cs.first_message,
                cs.last_message,
                COALESCE(cls.cluster_count, 0) as cluster_count,
                COALESCE(cls.clustered_messages, 0) as clustered_messages,
                COALESCE(cls.avg_cluster_size, 0.0) as avg_cluster_size,
                COALESCE(cls.max_cluster_size, 0) as max_cluster_size
            FROM chat_stats cs
            LEFT JOIN latest_chat_names lcn ON cs.chat_id = lcn.chat_id
            LEFT JOIN cluster_stats cls ON cs.chat_id = cls.chat_id
            ORDER BY cs.message_count DESC
            """

            result = self.db.db.execute(sql).arrow()
            return pl.DataFrame(pl.from_arrow(result))

        except Exception as e:
            print(f"Error getting chats list: {e}")
            return pl.DataFrame()

    def find_chat_by_name(self, chat_name: str) -> int | None:
        """Find chat ID by partial name match."""
        try:
            sql = f"""
            SELECT DISTINCT chat_id
            FROM {self.chat_names_table}
            WHERE LOWER(chat_name) LIKE LOWER(?)
            """
            result = self.db.db.execute(sql, [f"%{chat_name}%"]).fetchone()
            return result[0] if result else None
        except Exception:
            return None

    def get_large_clusters(self, chat_id: int | None = None, min_size: int = 10) -> pl.DataFrame | pl.Series:
        """Get all large clusters for a specific chat or all chats."""
        if not self._check_tables_exist():
            print(f"❌ No data found for phone {self.phone}")
            return pl.DataFrame()

        try:
            chat_filter = f"AND c.chat_id = {chat_id}" if chat_id else ""

            sql = f"""
            WITH latest_chat_names AS (
                SELECT chat_id, chat_name FROM (
                    SELECT chat_id, chat_name,
                           ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY COALESCE(last_seen, first_seen) DESC) AS rn
                    FROM {self.chat_names_table}
                ) t WHERE rn = 1
            ),
            cluster_sizes AS (
                SELECT
                    c.group_id,
                    c.chat_id,
                    COUNT(*) as size,
                    COUNT(DISTINCT m.from_id) as participant_count,
                    MIN(m.date) as start_time,
                    MAX(m.date) as end_time,
                    AVG(m.date) as avg_time
                FROM {self.clusters_table} c
                JOIN {self.messages_table} m ON c.message_id = m.message_id AND c.chat_id = m.chat_id
                WHERE 1=1 {chat_filter}
                GROUP BY c.group_id, c.chat_id
                HAVING COUNT(*) >= ?
            )
            SELECT
                cs.group_id,
                cs.chat_id,
                COALESCE(lcn.chat_name, 'Unknown Chat') as chat_name,
                cs.size as message_count,
                cs.participant_count,
                cs.start_time,
                cs.end_time,
                EXTRACT(EPOCH FROM (cs.end_time - cs.start_time)) / 3600.0 as duration_hours,
                cs.size / GREATEST(EXTRACT(EPOCH FROM (cs.end_time - cs.start_time)) / 3600.0, 0.1) as messages_per_hour
            FROM cluster_sizes cs
            LEFT JOIN latest_chat_names lcn ON cs.chat_id = lcn.chat_id
            ORDER BY cs.size DESC, cs.chat_id
            """

            result = self.db.db.execute(sql, [min_size]).arrow()
            df = pl.DataFrame(pl.from_arrow(result))

            # Format datetime columns and add intensity classification
            if len(df) > 0:
                df = df.with_columns(
                    [
                        pl.col("start_time").dt.strftime("%Y-%m-%d %H:%M").alias("start_time_str"),
                        pl.col("end_time").dt.strftime("%Y-%m-%d %H:%M").alias("end_time_str"),
                        pl.when(pl.col("messages_per_hour") >= 20)
                        .then(pl.lit("🔥 Very High"))
                        .when(pl.col("messages_per_hour") >= 10)
                        .then(pl.lit("🔴 High"))
                        .when(pl.col("messages_per_hour") >= 5)
                        .then(pl.lit("🟡 Medium"))
                        .otherwise(pl.lit("🟢 Low"))
                        .alias("intensity"),
                    ]
                )

            return df

        except Exception as e:
            print(f"Error getting large clusters: {e}")
            return pl.DataFrame()

    def analyze_cluster_details(self, chat_id: int, group_id: int) -> dict[str, Any]:
        """Get detailed analysis of a specific cluster."""
        try:
            # Get cluster messages with user names
            sql = f"""
            WITH latest_user_names AS (
                SELECT from_id, from_name FROM (
                    SELECT from_id, from_name,
                           ROW_NUMBER() OVER (PARTITION BY from_id ORDER BY COALESCE(last_seen, first_seen) DESC) AS rn
                    FROM {self.user_names_table}
                ) t WHERE rn = 1
            ),
            latest_chat_names AS (
                SELECT chat_id, chat_name FROM (
                    SELECT chat_id, chat_name,
                           ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY COALESCE(last_seen, first_seen) DESC) AS rn
                    FROM {self.chat_names_table}
                ) t WHERE rn = 1
            )
            SELECT
                m.message_id,
                m.chat_id,
                COALESCE(lcn.chat_name, 'Unknown Chat') as chat_name,
                m.text,
                m.from_id,
                COALESCE(lun.from_name, CAST(m.from_id AS VARCHAR)) as from_name,
                m.date,
                m.reply_to_message_id
            FROM {self.messages_table} m
            JOIN {self.clusters_table} c ON m.message_id = c.message_id AND m.chat_id = c.chat_id
            LEFT JOIN latest_user_names lun ON m.from_id = lun.from_id
            LEFT JOIN latest_chat_names lcn ON m.chat_id = lcn.chat_id
            WHERE c.group_id = ? AND c.chat_id = ?
            ORDER BY m.date
            """

            result = self.db.db.execute(sql, [group_id, chat_id]).arrow()
            messages_df = pl.DataFrame(pl.from_arrow(result))

            if len(messages_df) == 0:
                return {"error": "No messages found for this cluster"}

            # Calculate statistics
            stats = {
                "cluster_id": group_id,
                "chat_id": chat_id,
                "chat_name": messages_df["chat_name"][0],
                "message_count": len(messages_df),
                "participant_count": messages_df["from_id"].n_unique(),
                "start_time": messages_df["date"].min(),
                "end_time": messages_df["date"].max(),
                "messages": messages_df,
            }

            # Calculate duration and intensity
            duration = stats["end_time"] - stats["start_time"]
            duration_hours = duration.total_seconds() / 3600.0
            stats["duration_hours"] = duration_hours
            stats["messages_per_hour"] = stats["message_count"] / max(duration_hours, 0.1)

            # Participant breakdown
            participant_stats = (
                messages_df.group_by(["from_id", "from_name"])
                .agg(
                    [
                        pl.len().alias("message_count"),
                        pl.col("text").str.len_chars().mean().alias("avg_message_length"),
                    ]
                )
                .sort("message_count", descending=True)
            )
            stats["participants"] = participant_stats

            # Time-based analysis (messages per hour)
            time_analysis = (
                messages_df.with_columns(
                    [pl.col("date").dt.hour().alias("hour"), pl.col("date").dt.date().alias("date_only")]
                )
                .group_by(["date_only", "hour"])
                .agg([pl.len().alias("messages"), pl.col("from_id").n_unique().alias("active_users")])
                .sort(["date_only", "hour"])
            )
            stats["time_analysis"] = time_analysis

            # Find most intense period (1-hour window with most messages)
            if len(time_analysis) > 0:
                max_hour = time_analysis.sort("messages", descending=True).head(1)
                stats["peak_hour"] = {
                    "date": max_hour["date_only"][0],
                    "hour": max_hour["hour"][0],
                    "messages": max_hour["messages"][0],
                    "active_users": max_hour["active_users"][0],
                }

            return stats

        except Exception as e:
            return {"error": f"Error analyzing cluster: {e}"}

    def get_cluster_summary_data(self, chat_id: int, group_id: int) -> str:
        """Get cluster messages formatted for LLM summarization."""
        try:
            sql = f"""
            WITH latest_user_names AS (
                SELECT from_id, from_name FROM (
                    SELECT from_id, from_name,
                           ROW_NUMBER() OVER (PARTITION BY from_id ORDER BY COALESCE(last_seen, first_seen) DESC) AS rn
                    FROM {self.user_names_table}
                ) t WHERE rn = 1
            )
            SELECT
                m.date,
                COALESCE(lun.from_name, CAST(m.from_id AS VARCHAR)) as from_name,
                m.text
            FROM {self.messages_table} m
            JOIN {self.clusters_table} c ON m.message_id = c.message_id AND m.chat_id = c.chat_id
            LEFT JOIN latest_user_names lun ON m.from_id = lun.from_id
            WHERE c.group_id = ? AND c.chat_id = ?
            ORDER BY m.date
            """

            result = self.db.db.execute(sql, [group_id, chat_id]).arrow()
            messages_df = pl.DataFrame(pl.from_arrow(result))

            # Format messages for LLM
            formatted_messages = []
            for row in messages_df.iter_rows(named=True):
                timestamp = row["date"].strftime("%Y-%m-%d %H:%M:%S")
                from_name = row["from_name"]
                text = (row["text"] or "").strip()
                if text:
                    formatted_messages.append(f"[{timestamp}] {from_name}: {text}")

            return "\n".join(formatted_messages)

        except Exception as e:
            return f"Error getting cluster data: {e}"


def summarize_cluster_with_openai(cluster_text: str, api_key: str) -> str:
    """Summarize cluster using OpenAI API."""
    if not OPENAI_AVAILABLE:
        return "OpenAI library not available. Please install with: pip install openai"

    try:
        client = openai.OpenAI(api_key=api_key)

        prompt = f"""
        Analyze the following conversation cluster and provide:
        1. A brief summary of the main discussion
        2. Key topics and themes discussed
        3. Important moments or decisions made
        4. Participant dynamics and roles

        Conversation:
        {cluster_text}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using available model instead of GPT-5 nano
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing group conversations and extracting key insights.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.3,
        )

        return response.choices[0].message.content or "No summary generated"

    except Exception as e:
        return f"Error generating summary: {e}"


def extract_story_from_cluster(cluster_text: str, api_key: str, perspective: str = "third_person") -> str:
    """Extract a narrative story from cluster using OpenAI API."""
    if not OPENAI_AVAILABLE:
        return "OpenAI library not available. Please install with: pip install openai"

    try:
        client = openai.OpenAI(api_key=api_key)

        if perspective == "first_person":
            story_prompt = "Write a first-person narrative story based on the events and discussions in this conversation cluster. Focus on the main storyline and character experiences."
        else:
            story_prompt = "Write a third-person narrative story based on the events and discussions in this conversation cluster. Focus on the main storyline and character interactions."

        prompt = f"""
        {story_prompt}

        Conversation cluster:
        {cluster_text}

        Requirements:
        - Create a coherent narrative from the conversation
        - Include key events and character development
        - Maintain the essence of what was discussed
        - Make it engaging and readable
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a skilled storyteller who can transform conversations into engaging narratives.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
            temperature=0.7,
        )

        return response.choices[0].message.content or "No story generated"

    except Exception as e:
        return f"Error generating story: {e}"


def save_story_to_db(
    db: TelegramDatabase,
    phone: str,
    chat_id: int,
    group_id: int,
    story_text: str,
    tags: str,
    start_time: datetime,
    end_time: datetime,
    participants: list[str],
) -> bool:
    """Save extracted story to database."""
    try:
        phone_clean = phone.replace("+", "")
        stories_table = f"stories_{phone_clean}"

        # Create stories table if it doesn't exist
        db.db.execute(f"""
            CREATE TABLE IF NOT EXISTS {stories_table} (
                story_id INTEGER PRIMARY KEY,
                chat_id BIGINT,
                cluster_group_id INTEGER,
                associated_message_ids TEXT,
                story_text TEXT,
                tags TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                participants TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Get associated message IDs
        clusters_table = f"message_clusters_{phone_clean}"
        result = db.db.execute(
            f"SELECT message_id FROM {clusters_table} WHERE group_id = ? AND chat_id = ?", [group_id, chat_id]
        ).fetchall()

        message_ids = ",".join([str(row[0]) for row in result])
        participants_str = ",".join(participants)

        # Insert story
        db.db.execute(
            f"""
            INSERT INTO {stories_table}
            (chat_id, cluster_group_id, associated_message_ids, story_text, tags, start_time, end_time, participants)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [chat_id, group_id, message_ids, story_text, tags, start_time, end_time, participants_str],
        )

        return True

    except Exception as e:
        print(f"Error saving story to database: {e}")
        return False


def display_chats_table(df: pl.DataFrame) -> None:
    """Display chats in a user-friendly table format."""
    if len(df) == 0:
        print("No chats found.")
        return

    print("\n📋 Available Chats:")
    print("=" * 100)

    display_df = df.select(
        ["chat_id", "chat_name", "message_count", "participant_count", "cluster_count", "max_cluster_size"]
    ).with_columns(
        [
            pl.col("chat_name").str.slice(0, 30).alias("chat_name_short"),
        ]
    )

    print(display_df.to_pandas().to_string(index=False, max_colwidth=30))
    print()


def display_clusters_table(df: pl.DataFrame) -> None:
    """Display clusters in a user-friendly table format."""
    if len(df) == 0:
        print("No large clusters found.")
        return

    print("\n🔍 Large Clusters Found:")
    print("=" * 120)

    display_df = df.select(
        [
            "group_id",
            "chat_name",
            "message_count",
            "participant_count",
            "start_time_str",
            "duration_hours",
            "messages_per_hour",
            "intensity",
        ]
    ).with_columns(
        [
            pl.col("chat_name").str.slice(0, 25).alias("chat_short"),
            pl.col("duration_hours").round(2),
            pl.col("messages_per_hour").round(1),
        ]
    )

    print(display_df.to_pandas().to_string(index=False, max_colwidth=25))
    print()


def display_cluster_analysis(stats: dict[str, Any]) -> None:
    """Display detailed cluster analysis."""
    if "error" in stats:
        print(f"❌ {stats['error']}")
        return

    print(f"\n📊 Cluster Analysis - Group {stats['cluster_id']} in {stats['chat_name']}")
    print("=" * 80)

    print(f"📝 Messages: {stats['message_count']}")
    print(f"👥 Participants: {stats['participant_count']}")
    print(f"⏰ Duration: {stats['duration_hours']:.2f} hours")
    print(f"🚀 Intensity: {stats['messages_per_hour']:.1f} messages/hour")
    print(f"🕐 Period: {stats['start_time']} → {stats['end_time']}")

    if "peak_hour" in stats:
        peak = stats["peak_hour"]
        print(f"🔥 Peak hour: {peak['date']} at {peak['hour']:02d}:00 ({peak['messages']} messages)")

    print("\n👥 Participant Breakdown:")
    print(stats["participants"].to_pandas().to_string(index=False))

    if len(stats["time_analysis"]) > 0:
        print("\n⏰ Hourly Activity (showing top 10):")
        top_hours = stats["time_analysis"].sort("messages", descending=True).head(10)
        print(top_hours.to_pandas().to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Terrorblade Cluster Analysis CLI")
    parser.add_argument("--phone", required=True, help="Phone number (e.g., +1234567890)")
    parser.add_argument("--db-path", default="auto", help="Database path (default: auto)")
    parser.add_argument("--min-size", type=int, default=10, help="Minimum cluster size (default: 10)")

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = ClusterAnalyzer(args.phone, args.db_path)

    try:
        while True:
            print("\n🛠️  Terrorblade Cluster Analysis Tool")
            print("=" * 50)
            print("1. List all chats")
            print("2. Show large clusters (all chats)")
            print("3. Show large clusters for specific chat")
            print("4. Analyze specific cluster")
            print("5. Summarize cluster with AI")
            print("6. Extract story from cluster")
            print("7. Exit")

            choice = input("\nSelect option (1-7): ").strip()

            if choice == "1":
                chats_df = analyzer.get_chats_list()
                display_chats_table(chats_df)

            elif choice == "2":
                clusters_df = analyzer.get_large_clusters(min_size=args.min_size)
                display_clusters_table(pl.DataFrame(clusters_df))

            elif choice == "3":
                chat_input = input("Enter chat name (partial) or chat ID: ").strip()

                try:
                    chat_id = int(chat_input)
                except ValueError:
                    chat_id = analyzer.find_chat_by_name(chat_input) # type: ignore
                    if not chat_id:
                        print(f"❌ No chat found matching '{chat_input}'")
                        continue

                clusters_df = analyzer.get_large_clusters(chat_id=chat_id, min_size=args.min_size)
                display_clusters_table(pl.DataFrame(clusters_df))

            elif choice == "4":
                try:
                    chat_id = int(input("Enter chat ID: ").strip())
                    group_id = int(input("Enter cluster group ID: ").strip())

                    stats = analyzer.analyze_cluster_details(chat_id, group_id)
                    display_cluster_analysis(stats)

                except ValueError:
                    print("❌ Please enter valid numeric IDs")

            elif choice == "5":
                if not OPENAI_AVAILABLE:
                    print("❌ OpenAI library not installed. Please install with: pip install openai")
                    continue

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    print("❌ OPENAI_API_KEY not found in environment variables")
                    continue

                try:
                    chat_id = int(input("Enter chat ID: ").strip())
                    group_id = int(input("Enter cluster group ID: ").strip())

                    print("🤖 Generating AI summary...")
                    cluster_text = analyzer.get_cluster_summary_data(chat_id, group_id)
                    summary = summarize_cluster_with_openai(cluster_text, api_key)

                    print("\n📄 AI Summary:")
                    print("=" * 60)
                    print(summary)

                    print("\n💾 Saving summary to database...")
                    stats = analyzer.analyze_cluster_details(chat_id, group_id)
                    if "error" not in stats:
                        participants = stats["participants"]["from_name"].to_list()
                        tags = "ai_summary,cluster_analysis"

                        success = save_story_to_db(
                            analyzer.db,
                            args.phone,
                            chat_id,
                            group_id,
                            summary,
                            tags,
                            stats["start_time"],
                            stats["end_time"],
                            participants,
                        )

                        if success:
                            print("✅ Summary saved to database successfully!")
                            print(f"📋 Details: {len(summary)} chars, {len(participants)} participants, tags: {tags}")
                        else:
                            print("❌ Failed to save summary to database")
                    else:
                        print("⚠️  Could not get cluster stats for database logging")

                except ValueError:
                    print("❌ Please enter valid numeric IDs")

            elif choice == "6":
                if not OPENAI_AVAILABLE:
                    print("❌ OpenAI library not installed. Please install with: pip install openai")
                    continue

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    print("❌ OPENAI_API_KEY not found in environment variables")
                    continue

                try:
                    chat_id = int(input("Enter chat ID: ").strip())
                    group_id = int(input("Enter cluster group ID: ").strip())

                    perspective = input("Story perspective (first_person/third_person) [third_person]: ").strip()
                    if not perspective:
                        perspective = "third_person"

                    print("📖 Generating story...")
                    cluster_text = analyzer.get_cluster_summary_data(chat_id, group_id)
                    story = extract_story_from_cluster(cluster_text, api_key, perspective)

                    print(f"\n📚 Generated Story ({perspective}):")
                    print("=" * 60)
                    print(story)

                    # Option to save to database
                    save_choice = input("\nSave story to database? (y/n): ").strip().lower()
                    if save_choice == "y":
                        tags = input("Enter tags (comma-separated): ").strip()

                        # Get cluster details for metadata
                        stats = analyzer.analyze_cluster_details(chat_id, group_id)
                        if "error" not in stats:
                            participants = stats["participants"]["from_name"].to_list()
                            success = save_story_to_db(
                                analyzer.db,
                                args.phone,
                                chat_id,
                                group_id,
                                story,
                                tags,
                                stats["start_time"],
                                stats["end_time"],
                                participants,
                            )
                            if success:
                                print("✅ Story saved to database!")
                            else:
                                print("❌ Failed to save story")

                except ValueError:
                    print("❌ Please enter valid numeric IDs")

            elif choice == "7":
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice. Please select 1-7.")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
