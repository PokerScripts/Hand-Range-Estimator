#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Оценка диапазона оппонента по линии действий в раздаче.
CLI-инструмент: анализирует последовательность действий и
строит вероятностный диапазон рук (Texas Hold'em NL).
"""

import argparse
import json
import re
import itertools
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

RANKS = "AKQJT98765432"
SUITS = "shdc"


# ==============================================================
# ----------------------- ВСПОМОГАТЕЛЬНЫЕ ----------------------
# ==============================================================

def all_combos_for_hand(hand: str) -> List[str]:
    """Разворачивает обозначение руки (например 'AKs', 'AKo', '77')
       в конкретные комбы (например ['AsKs','AhKh',...]).
    """
    combos = []
    if len(hand) == 2:  # Пара
        r1 = r2 = hand[0]
        for s1, s2 in itertools.combinations(SUITS, 2):
            combos.append(r1 + s1 + r2 + s2)
    elif len(hand) == 3 and hand[2] == 's':
        r1, r2 = hand[0], hand[1]
        for s in SUITS:
            combos.append(r1 + s + r2 + s)
    elif len(hand) == 3 and hand[2] == 'o':
        r1, r2 = hand[0], hand[1]
        for s1 in SUITS:
            for s2 in SUITS:
                if s1 != s2:
                    combos.append(r1 + s1 + r2 + s2)
    else:
        raise ValueError(f"Неверный формат руки: {hand}")
    return combos


def parse_range_notation(rng: str) -> List[str]:
    """Парсит диапазон вроде '22+,A2s+,A7o+,K9s+,KTo+,QTs+,QJo,JTs,T9s,98s,87s'."""
    result = []

    def rank_index(r):
        return RANKS.index(r)

    tokens = [t.strip() for t in rng.split(',') if t.strip()]
    for token in tokens:
        # Префлоп диапазоны со знаком +
        m = re.match(r"^([2-9TJQKA]{1,2})([so]?)\+$", token)
        if m:
            base = m.group(1)
            suit = m.group(2)
            if len(base) == 2:  # например "A2"
                high, low = base[0], base[1]
                for r in RANKS[rank_index(low):-1]:
                    result.append(high + r + suit)
            elif len(base) == 1:  # пары вроде "22+"
                low = base
                for r in RANKS[rank_index(low)::-1]:
                    if RANKS.index(r) <= RANKS.index('A'):
                        result.append(r + r)
        else:
            result.append(token)
    return result


def normalize_weights(d: Dict[str, float]) -> Dict[str, float]:
    """Нормализация весов до суммы 1."""
    total = sum(d.values())
    if total == 0:
        return d
    for k in d:
        d[k] /= total
    return d


# ==============================================================
# ----------------------- МОДЕЛЬ / RANGE ------------------------
# ==============================================================

def init_preflop_range(model: Dict[str, Any], villain_pos: str, vs_pos: str) -> Dict[str, float]:
    """Создаёт базовый диапазон префлопа из модели.
       Возвращает dict{combo: weight}.
    """
    range_dict = {}
    # Ищем подходящий раздел в модели
    try:
        node = model["preflop"]["open"].get(villain_pos)
        if node:
            base_range = parse_range_notation(node["range"])
            w = node.get("weight", 1.0)
            for hand in base_range:
                for c in all_combos_for_hand(hand):
                    range_dict[c] = w
    except Exception as e:
        print(f"[!] Ошибка загрузки диапазона из модели: {e}")

    return normalize_weights(range_dict)


# ==============================================================
# ----------------------- ПАРСЕР ЛИНИИ -------------------------
# ==============================================================

class Event:
    def __init__(self, street: str, actor: str, action: str, size: float = None):
        self.street = street
        self.actor = actor
        self.action = action
        self.size = size  # в процентах или bb


def parse_line(line: str) -> List[Event]:
    """Парсит строку --line в список Event."""
    events = []
    current_street = "PREFLOP"

    tokens = [t.strip() for t in line.split(';') if t.strip()]
    for t in tokens:
        # Улицы
        m = re.match(r"^(FLOP|TURN|RIVER)\s+(.+)$", t, re.I)
        if m:
            current_street = m.group(1).upper()
            continue

        # Действие
        m = re.match(r"^([A-Z]{2,3}):?[\d\.]*bb?\s*(.*)$", t.strip(), re.I)
        if m:
            actor = m.group(1).upper()
            act = m.group(2)
            size = None
            if m2 := re.search(r"(\d+(\.\d+)?)%?", act):
                size_val = m2.group(1)
                size = float(size_val)
                act = act.replace(size_val, "").strip()
            events.append(Event(current_street, actor, act.lower(), size))
    return events


# ==============================================================
# ----------------------- ОБНОВЛЕНИЕ RANGE ---------------------
# ==============================================================

def update_range(range_dict: Dict[str, float], event: Event, model: Dict[str, Any],
                 context: Dict[str, Any]) -> Dict[str, float]:
    """Обновляет веса диапазона по действию (очень упрощённая эвристика)."""
    multipliers = {}

    # Простая логика: если оппонент ставит маленький бет - повышаем вес слабых рук,
    # если большой - сильных.
    if event.action.startswith("bet") or event.action.startswith("cbet"):
        if event.size and event.size <= 40:
            for combo in range_dict:
                if any(x in combo for x in ["7", "6", "5", "4", "3", "2"]):
                    multipliers[combo] = 1.2
                else:
                    multipliers[combo] = 0.9
        elif event.size and event.size >= 60:
            for combo in range_dict:
                if any(x in combo for x in ["A", "K", "Q", "J", "T"]):
                    multipliers[combo] = 1.2
                else:
                    multipliers[combo] = 0.8
        else:
            multipliers = {c: 1.0 for c in range_dict}

    elif "fold" in event.action:
        # Если фолд — просто обнуляем веса
        for combo in range_dict:
            range_dict[combo] = 0.0
        return range_dict

    else:
        # По умолчанию ничего не меняем
        multipliers = {c: 1.0 for c in range_dict}

    # Применяем множители
    for c in range_dict:
        range_dict[c] *= multipliers.get(c, 1.0)

    return normalize_weights(range_dict)


# ==============================================================
# ----------------------- АГРЕГАЦИЯ -----------------------------
# ==============================================================

def aggregate_by_hand(range_dict: Dict[str, float]) -> Dict[str, float]:
    """Агрегирует веса по обозначениям типа AKs, AKo, 77."""
    agg = defaultdict(float)
    for combo, w in range_dict.items():
        r1, s1, r2, s2 = combo[0], combo[1], combo[2], combo[3]
        if r1 == r2:
            key = r1 + r2
        elif s1 == s2:
            key = r1 + r2 + "s"
        else:
            key = r1 + r2 + "o"
        # нормализуем порядок (A>K, и т.д.)
        if RANKS.index(r1) > RANKS.index(r2):
            key = r1 + r2 + key[-1] if len(key) == 3 else r1 + r2
        else:
            key = r2 + r1 + key[-1] if len(key) == 3 else r2 + r1
        agg[key] += w
    return normalize_weights(agg)


def render_matrix(agg: Dict[str, float]) -> str:
    """Рисует 13x13 матрицу (верхний треугольник оффсьют, нижний — сьютед, диагональ — пары)."""
    grid = [["" for _ in range(13)] for _ in range(13)]
    rank_list = list(RANKS)
    for i, r1 in enumerate(rank_list):
        for j, r2 in enumerate(rank_list):
            key = None
            if i < j:
                key = r1 + r2 + "s"
            elif i > j:
                key = r2 + r1 + "o"
            else:
                key = r1 + r2
            val = agg.get(key, 0)
            if val == 0:
                ch = " . "
            elif val < 0.005:
                ch = " · "
            elif val < 0.01:
                ch = " + "
            elif val < 0.02:
                ch = " * "
            elif val < 0.05:
                ch = " # "
            else:
                ch = "███"
            grid[i][j] = ch
    # Форматируем
    s = "     " + " ".join(rank_list) + "\n"
    for i, r1 in enumerate(rank_list):
        s += f"{r1}  " + " ".join(grid[i]) + "\n"
    return s


def report_top_hands(agg: Dict[str, float], top_pct: float = 20.0) -> List[Tuple[str, float]]:
    """Возвращает топ N% рук по весу."""
    sorted_items = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    cumulative = 0.0
    res = []
    for h, w in sorted_items:
        cumulative += w * 100
        res.append((h, w))
        if cumulative >= top_pct:
            break
    return res


# ==============================================================
# ----------------------- ОСНОВНОЙ MAIN ------------------------
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description="Оценка диапазона оппонента по линии действий (estimator.py)")
    parser.add_argument("--line", required=True, help="Строка с описанием раздачи (DSL)")
    parser.add_argument("--villain-pos", required=True, help="Позиция оппонента (например SB)")
    parser.add_argument("--hero-pos", required=False, help="Позиция героя (например BTN)")
    parser.add_argument("--model", default="model.example.json", help="JSON-модель диапазонов")
    parser.add_argument("--top", type=float, default=20.0, help="Процент топ-рук для вывода")
    parser.add_argument("--output", choices=["txt", "json", "csv"], default="txt")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Загружаем модель
    try:
        with open(args.model, "r", encoding="utf-8") as f:
            model = json.load(f)
    except Exception as e:
        print(f"[!] Не удалось загрузить модель {args.model}: {e}")
        model = {}

    # Парсим линию
    events = parse_line(args.line)
    if args.debug:
        for e in events:
            print(f"{e.street}: {e.actor} {e.action} size={e.size}")

    # Инициализация диапазона
    rng = init_preflop_range(model, args.villain-pos, args.hero_pos or "BTN")
    if not rng:
        print("[!] Не удалось инициализировать диапазон, возможно отсутствует модель для позиции.")
        return

    context = {"street": "PREFLOP"}
    # Апдейтим по событиям
    for ev in events:
        if ev.actor == args.villain_pos:
            rng = update_range(rng, ev, model, context)
            context["street"] = ev.street
            if args.debug:
                print(f"\n>>> После {ev.action} {ev.size or ''}% на {ev.street}")
                top = report_top_hands(aggregate_by_hand(rng), 10)
                print("Top-руки:", [h for h, _ in top[:5]])

    # Агрегируем
    agg = aggregate_by_hand(rng)
    top = report_top_hands(agg, args.top)

    if args.output == "txt":
        print("\n=== СВОДКА ===")
        print(f"Рук в диапазоне: {len(agg)}")
        print(f"Top {args.top}% рук:")
        for h, w in top:
            print(f"{h:5s}  {w:.4f}")
        print("\nМатрица диапазона:")
        print(render_matrix(agg))

    elif args.output == "json":
        print(json.dumps(agg, indent=2, ensure_ascii=False))

    elif args.output == "csv":
        print("hand,weight")
        for h, w in sorted(agg.items(), key=lambda x: x[1], reverse=True):
            print(f"{h},{w:.6f}")


if __name__ == "__main__":
    main()