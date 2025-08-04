#!/usr/bin/env python3
"""
Animal Message Transformer Script

This script transforms Telegram message data by:
1. Replacing author names with animal-themed names
2. Converting all text messages to discuss animals and how cool they are
3. Maintaining the same structure and messaging order
"""

import argparse
import json
import os
import random
from typing import Any

# Animal-themed author name replacements
ANIMAL_AUTHORS = {
    "Seva ✨": "Lion 🦁",
    "А👑": "Eagle 🦅",
    "🐈Аня": "Tiger 🐯",
    "Маша Балуева 🌮": "Dolphin 🐬",
    "Поддубка🙃Карлагач": "Penguin 🐧",
    "POKRAS LAMPAS ®": "Wolf 🐺",
    "Арина Стамбул": "Fox 🦊",
    "Арина": "Fox 🦊",
}

# Animal topics and phrases for message transformation
ANIMAL_TOPICS = [
    "Lions are absolutely amazing! Their majestic manes and powerful roars make them the kings of the jungle.",
    "Elephants are incredible creatures. Their intelligence and strong family bonds are truly inspiring.",
    "Dolphins are so cool! They're incredibly smart and playful, always swimming with such grace.",
    "Tigers are magnificent! Their beautiful stripes and stealth hunting skills are just awesome.",
    "Penguins are adorable! The way they waddle and slide on ice is absolutely charming.",
    "Wolves are fascinating! Their pack mentality and howling communication is really cool.",
    "Giraffes are amazing! Their long necks and gentle nature make them so unique.",
    "Pandas are the cutest! Their black and white fur and bamboo eating habits are adorable.",
    "Sharks are incredible! Their power and swimming abilities are truly impressive.",
    "Bears are awesome! Their strength and forest habitat are really cool.",
    "Eagles are majestic! Their sharp vision and soaring flight are absolutely amazing.",
    "Monkeys are so fun! Their playful nature and climbing skills are really cool.",
    "Zebras are beautiful! Their unique stripes and social behavior are fascinating.",
    "Kangaroos are amazing! Their hopping movement and pouches are really cool.",
    "Koalas are adorable! Their eucalyptus diet and tree-climbing are so cute.",
    "Cheetahs are incredible! Their speed and hunting skills are absolutely amazing.",
    "Gorillas are fascinating! Their strength and family bonds are really cool.",
    "Rhinos are awesome! Their thick skin and horn are truly impressive.",
    "Hippos are amazing! Their size and water habitat are really cool.",
    "Crocodiles are incredible! Their ancient appearance and hunting skills are fascinating.",
]

# Short animal responses for quick messages
SHORT_ANIMAL_RESPONSES = [
    "So cool!",
    "Amazing!",
    "Incredible!",
    "Awesome!",
    "Fascinating!",
    "Beautiful!",
    "Wonderful!",
    "Fantastic!",
    "Brilliant!",
    "Stunning!",
    "Magnificent!",
    "Gorgeous!",
    "Spectacular!",
    "Marvelous!",
    "Extraordinary!",
]

# Animal-related questions
ANIMAL_QUESTIONS = [
    "Did you know how amazing animals are?",
    "Have you ever thought about how cool animals are?",
    "Isn't it fascinating how animals behave?",
    "What's your favorite animal?",
    "How cool are animals, right?",
    "Animals are just incredible, aren't they?",
    "Have you seen how amazing animals can be?",
    "What do you think about animals?",
    "Isn't nature just wonderful?",
    "How awesome are animals?",
]


def get_random_animal_message(original_length: int) -> str:
    """Generate an animal-themed message based on original message length."""
    if original_length < 10:
        return random.choice(SHORT_ANIMAL_RESPONSES)
    elif original_length < 50:
        return random.choice(ANIMAL_QUESTIONS)
    else:
        return random.choice(ANIMAL_TOPICS)


def transform_message_text(text: str) -> str:
    """Transform message text to animal-themed content while preserving some structure."""
    # Handle case where text might be None or empty
    if not text or (isinstance(text, str) and text.strip() == ""):
        return get_random_animal_message(10)

    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)

    # Preserve some punctuation and structure
    original_length = len(text)

    # If it's a question, make it an animal question
    if text.endswith("?"):
        return random.choice(ANIMAL_QUESTIONS)

    # If it's a short response, make it a short animal response
    if original_length < 20:
        return random.choice(SHORT_ANIMAL_RESPONSES)

    # For longer messages, use detailed animal topics
    return get_random_animal_message(original_length)


def transform_author_name(author: str) -> str:
    """Transform author name to animal-themed name."""
    return ANIMAL_AUTHORS.get(
        author,
        f"Animal {random.choice(['🦁', '🦅', '🐯', '🐬', '🐧', '🐺', '🦒', '🐼', '🦈', '🐻'])}",
    )


def transform_messages(data: dict[str, Any]) -> dict[str, Any]:
    """Transform all messages in the data structure."""
    if "chats" in data and "list" in data["chats"]:
        for chat in data["chats"]["list"]:
            if "messages" in chat:
                for message in chat["messages"]:
                    # Transform author name
                    if "from" in message:
                        message["from"] = transform_author_name(message["from"])

                    # Transform text content
                    if "text" in message and message["text"]:
                        message["text"] = transform_message_text(message["text"])

                        # Update text_entities if they exist
                        if "text_entities" in message and message["text_entities"]:
                            for entity in message["text_entities"]:
                                if entity.get("type") == "plain":
                                    entity["text"] = message["text"]

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Transform Telegram messages to animal-themed content"
    )
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("output_file", help="Path to the output JSON file")
    parser.add_argument(
        "--backup", action="store_true", help="Create a backup of the original file"
    )

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return

    # Create backup if requested
    if args.backup:
        backup_file = f"{args.input_file}.backup"
        try:
            with open(args.input_file, encoding="utf-8") as f:
                backup_data = f.read()
            with open(backup_file, "w", encoding="utf-8") as f:
                f.write(backup_data)
            print(f"Backup created: {backup_file}")
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")

    # Read and transform the data
    try:
        with open(args.input_file, encoding="utf-8") as f:
            data = json.load(f)

        print("Transforming messages...")
        transformed_data = transform_messages(data)

        # Write the transformed data
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(transformed_data, f, ensure_ascii=False, indent=1)

        print(f"Transformation complete! Output saved to: {args.output_file}")

        # Print some statistics
        message_count = 0
        author_count = 0
        if "chats" in transformed_data and "list" in transformed_data["chats"]:
            for chat in transformed_data["chats"]["list"]:
                if "messages" in chat:
                    message_count += len(chat["messages"])
                    authors = set()
                    for message in chat["messages"]:
                        if "from" in message:
                            authors.add(message["from"])
                    author_count += len(authors)

        print(f"Transformed {message_count} messages with {author_count} unique animal authors")

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
