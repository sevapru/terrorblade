import json
from typing import Dict, List, Tuple

import polars as pl
from vllm import LLM, SamplingParams

from terrorblade.data.dtypes import base_emotions, dialogue_categories
from terrorblade.data.preprocessing.TextPreprocessor import TextPreprocessor


class DialogueAnalyzer(TextPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            self.llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", trust_remote_code=True)
        except Exception as e:
            print(f"Warning: vLLM initialization error: {e}")
            print("Falling back to CPU execution")
            self.llm = LLM(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                trust_remote_code=True,
                dtype="float32",
                gpu_memory_utilization=0.0,
            )

        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)

    def classify_topic_hierarchy(self, text: str) -> Dict[str, str]:
        """
        Определяет основную категорию и подкатегорию текста из предопределенного списка.

        Args:
            text (str): Текст для классификации

        Returns:
            Dict[str, str]: Словарь с категорией и подкатегорией
        """
        categories_str = json.dumps(dialogue_categories, ensure_ascii=False)
        prompt = f"""Classify the following text into main category and subcategory.
                     Use only categories from this list:
                     {categories_str}

                    Text:
                    {text}

Return JSON in format: {{"category": "main_category", "subcategory": "subcategory"}}"""

        outputs = self.llm.generate([prompt], self.sampling_params)
        try:
            result = json.loads(outputs[0].outputs[0].text.strip())
            return result
        except Exception as e:
            print(f"Warning: topic classification error: {e}")
            return {"category": "personal", "subcategory": "other"}

    def analyze_emotions(self, text: str) -> Dict[str, float]:
        """
        Анализирует текст по 7 базовым эмоциям.

        Args:
            text (str): Текст для анализа

        Returns:
            Dict[str, float]: Словарь с оценками по каждой эмоции
        """
        emotions_str = json.dumps(base_emotions, ensure_ascii=False)
        prompt = f"""Analyze the emotional content of the following text.
        Rate each emotion from 0.0 to 1.0 based on these basic emotions:
        {emotions_str}

        Text:
        {text}

        Return only JSON with ratings without explanations."""

        outputs = self.llm.generate([prompt], self.sampling_params)
        try:
            return json.loads(outputs[0].outputs[0].text.strip())
        except Exception as e:
            print(f"Warning: emotion analysis error: {e}")
            return {emotion: 0.0 for emotion in base_emotions.keys()}

    def get_topic_distribution(self, df: pl.DataFrame) -> Dict[str, Dict[str, int]]:
        """
        Анализирует распределение тем по категориям во всем диалоге.

        Args:
            df (pl.DataFrame): DataFrame с обработанными сообщениями

        Returns:
            Dict[str, Dict[str, int]]: Словарь с количеством сообщений по категориям и подкатегориям
        """
        topic_dist = {cat: {subcat: 0 for subcat in subcats} for cat, subcats in dialogue_categories.items()}

        for group in df.group_by("group"):
            text = " ".join(group.get_column("text"))
            classification = self.classify_topic_hierarchy(text)
            topic_dist[classification["category"]][classification["subcategory"]] += 1

        return topic_dist

    def get_dominant_topics(self, df: pl.DataFrame, top_n: int = 5) -> List[Tuple[str, str, int]]:
        """
        Определяет наиболее часто встречающиеся темы в диалоге.

        Args:
            df (pl.DataFrame): DataFrame с обработанными сообщениями
            top_n (int): Количество топовых тем для возврата

        Returns:
            List[Tuple[str, str, int]]: Список кортежей (категория, подкатегория, количество)
        """
        topic_dist = self.get_topic_distribution(df)
        flat_topics = []

        for category, subcats in topic_dist.items():
            for subcat, count in subcats.items():
                flat_topics.append((category, subcat, count))

        return sorted(flat_topics, key=lambda x: x[2], reverse=True)[:top_n]

    def process_dialogue(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Обрабатывает диалог, добавляя анализ тем и эмоций.

        Args:
            df (pl.DataFrame): Исходный DataFrame

        Returns:
            pl.DataFrame: Обработанный DataFrame с добавленными колонками
        """
        # Обработка базовым препроцессором
        df = super().process_message_groups(df)

        # Добавляем анализ для каждой группы
        topics = []
        emotions = []

        for group in df.group_by("group"):
            text = " ".join(group.get_column("text"))

            topic_class = self.classify_topic_hierarchy(text)
            emotion_scores = self.analyze_emotions(text)

            topics.append(topic_class)
            emotions.append(emotion_scores)

        df = df.with_columns(
            [
                pl.Series("topic_classification", topics),
                pl.Series("emotion_scores", emotions),
            ]
        )

        return df

    def generate_dialogue_summary(self, df: pl.DataFrame) -> Dict:
        """
        Генерирует общее саммари диалога с распределением тем и эмоций.

        Args:
            df (pl.DataFrame): Обработанный DataFrame

        Returns:
            Dict: Словарь с обобщенной информацией о диалоге
        """
        topic_dist = self.get_topic_distribution(df)
        dominant_topics = self.get_dominant_topics(df)

        # Вычисляем средние эмоциональные оценки
        emotion_scores = df.get_column("emotion_scores")
        avg_emotions = {
            emotion: sum(score[emotion] for score in emotion_scores) / len(emotion_scores)
            for emotion in base_emotions.keys()
        }

        return {
            "topic_distribution": topic_dist,
            "dominant_topics": dominant_topics,
            "average_emotions": avg_emotions,
            "message_count": len(df),
            "group_count": df.get_column("group").n_unique(),
        }
