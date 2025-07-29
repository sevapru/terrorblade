import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from thoth.analyzer import ThothAnalyzer

load_dotenv(Path().parent.parent / ".env")
# Or specify configuration directly
analyzer = ThothAnalyzer(db_path=os.getenv("DUCKDB_PATH"), phone="+31627866359")

# Import data from DuckDB
analyzer.import_from_duckdb()

# Analyze topics
topics = analyzer.analyze_topics(
    chat_id=123456789, n_topics=5, date_from=datetime.now() - timedelta(days=30)
)

# Analyze topic sentiment
sentiment = analyzer.analyze_topic_sentiment(topic_id=1)

# Search for semantically similar messages
results = analyzer.search(
    query="What do you think about the new policy?", chat_id=123456789, limit=5
)


# From text processor
def display_dialog_chunks(self, embeddings=None):
    if embeddings is None:
        if self.embeddings is None:
            raise ValueError("No embeddings available. Please calculate embeddings first.")
        # Create a copy to avoid holding references to the original tensor
        embeddings = self.embeddings.clone().cpu()
    elif embeddings.is_cuda:
        # Work with CPU copy to avoid GPU memory issues
        embeddings = embeddings.cpu()

    distances = self.calculate_sliding_distances(embeddings)

    # Initial threshold value
    init_threshold = 99  # Starting at 99%

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)
    y_upper_bound = 0.2

    # Plot function
    def plot_chunks(breakpoint_percentile_threshold):
        ax.clear()
        # Convert tensor to numpy for plotting
        distances_np = distances.numpy() if torch.is_tensor(distances) else distances
        sns.lineplot(x=range(len(distances_np)), y=distances_np, label="Cosine Distance", ax=ax)
        ax.set_ylim(0, y_upper_bound)
        ax.set_xlim(0, len(distances_np))

        breakpoint_distance_threshold = np.percentile(distances_np, breakpoint_percentile_threshold)
        ax.axhline(
            y=breakpoint_distance_threshold,
            color="r",
            linestyle="-",
            label="Threshold",
        )

        num_distances_above_threshold = np.sum(distances_np > breakpoint_distance_threshold)
        ax.text(
            len(distances_np) * 0.01,
            y_upper_bound / 50,
            f"{num_distances_above_threshold + 1} Chunks",
        )

        indices_above_thresh = np.where(distances_np > breakpoint_distance_threshold)[0]
        boundaries = [0] + (indices_above_thresh + 1).tolist() + [embeddings.shape[0]]

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            ax.axvspan(start_idx, end_idx, facecolor=plt.cm.get_cmap("tab10")(i % 10), alpha=0.25)
            ax.text(
                x=(start_idx + end_idx) / 2,
                y=breakpoint_distance_threshold + y_upper_bound / 20,
                s=f"Chunk #{i+1}",
                horizontalalignment="center",
                rotation="vertical",
            )

        ax.set_title("Dialog Chunks Based on Embedding Breakpoints")
        ax.set_xlabel("Message Index")
        ax.set_ylabel("Cosine Distance")
        ax.legend()

    # Initial plot
    plot_chunks(init_threshold)

    # Slider
    axcolor = "lightgoldenrodyellow"
    ax_thresh = plt.axes((0.15, 0.1, 0.65, 0.03), facecolor=axcolor)
    thresh_slider = Slider(
        ax=ax_thresh,
        label="Threshold (%)",
        valmin=0,
        valmax=100,
        valinit=init_threshold,
        valstep=1,
    )

    # Update function
    def update(val):
        plot_chunks(thresh_slider.val)
        fig.canvas.draw_idle()

    thresh_slider.on_changed(update)

    plt.show()
