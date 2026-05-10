"""
LLMCouncil — confronte 6 LLMs (DeepSeek, Grok, Kimi, OpenRouter, Gemini, OpenAI)
pour chaque signal et produit un verdict de consensus pondéré.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Tu es un expert en trading crypto spécialisé dans les tendances haussières.
Analyse cette opportunité de trading et réponds UNIQUEMENT en JSON valide :

Paire : {symbol}
Stratégie suggérée : {strategy}
Score technique CENTINA : {score}/100
Régime de marché : {regime}
RSI : {rsi}
Volume spike : {volume_spike}x
EMA alignées : {ema_aligned}
News sentiment : {news_sentiment}/100
Durée estimée : {duration_h}h
ATR : {atr_pct}%

Question : Dois-je acheter cette paire maintenant pour capturer +5% à +7% dans les {duration_h} prochaines heures ?

Réponds UNIQUEMENT avec ce JSON (rien d'autre) :
{{"verdict":"BUY","confidence":75,"reasoning":"EMA alignées, volume fort, momentum haussier","risk_level":"MEDIUM","estimated_gain_pct":5.5}}"""


class LLMCouncil:
    """
    Interroge 4 LLMs gratuits en parallèle pour chaque signal.
    Vote pondéré → verdict consensus + bonus/malus score CENTINA.
    """

    def __init__(self):
        self._weights = {
            "deepseek": 1.0, "grok": 1.2, "kimi": 1.0,
            "openrouter": 1.0, "gemini": 1.2, "openai": 1.3,
        }
        self._accuracy: dict[str, list[int]] = {
            k: [] for k in self._weights
        }
        self._enabled = any([
            os.getenv("DEEPSEEK_API_KEY"),
            os.getenv("GROK_API_KEY"),
            os.getenv("KIMI_API_KEY"),
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("GEMINI_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
        ])

    async def consult(self, opportunity_context: dict) -> dict:
        """Interroge tous les LLMs en parallèle. Retourne le consensus."""
        if not self._enabled:
            return self._empty_consensus()

        ctx = {
            "symbol":       opportunity_context.get("symbol", "?"),
            "strategy":     opportunity_context.get("strategy", "B"),
            "score":        opportunity_context.get("score", 0),
            "regime":       opportunity_context.get("regime", "TRENDING"),
            "rsi":          round(opportunity_context.get("rsi", 50), 1),
            "volume_spike": round(opportunity_context.get("volume_spike", 1.0), 2),
            "ema_aligned":  opportunity_context.get("ema_aligned", True),
            "news_sentiment": opportunity_context.get("news_sentiment", 50),
            "duration_h":   round(opportunity_context.get("duration_h", 3), 1),
            "atr_pct":      round(opportunity_context.get("atr_pct", 2.0), 2),
        }
        prompt = PROMPT_TEMPLATE.format(**ctx)

        responses = await asyncio.gather(
            self._ask_deepseek(prompt),
            self._ask_grok(prompt),
            self._ask_kimi(prompt),
            self._ask_openrouter(prompt),
            self._ask_gemini(prompt),
            self._ask_openai(prompt),
            return_exceptions=True,
        )
        return self._compute_consensus(list(responses))

    # ── LLM callers ───────────────────────────────────────────────────

    async def _ask_deepseek(self, prompt: str) -> dict:
        key = os.getenv("DEEPSEEK_API_KEY", "")
        if not key:
            return {"error": "no_key", "model": "deepseek"}
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                r = await s.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    json={"model": "deepseek-chat", "max_tokens": 200,
                          "messages": [{"role": "user", "content": prompt}]},
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=aiohttp.ClientTimeout(total=10),
                )
                data = await r.json()
                text = data["choices"][0]["message"]["content"]
                return {**_parse_json(text), "model": "deepseek"}
        except Exception as e:
            logger.debug("DeepSeek error: %s", e)
            return {"error": str(e), "model": "deepseek"}

    async def _ask_grok(self, prompt: str) -> dict:
        key = os.getenv("GROK_API_KEY", "")
        if not key:
            return {"error": "no_key", "model": "grok"}
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                r = await s.post(
                    "https://api.x.ai/v1/chat/completions",
                    json={"model": "grok-3-mini", "max_tokens": 200,
                          "messages": [{"role": "user", "content": prompt}]},
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=aiohttp.ClientTimeout(total=10),
                )
                data = await r.json()
                text = data["choices"][0]["message"]["content"]
                return {**_parse_json(text), "model": "grok"}
        except Exception as e:
            logger.debug("Grok error: %s", e)
            return {"error": str(e), "model": "grok"}

    async def _ask_kimi(self, prompt: str) -> dict:
        key = os.getenv("KIMI_API_KEY", "")
        if not key:
            return {"error": "no_key", "model": "kimi"}
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                r = await s.post(
                    "https://api.moonshot.cn/v1/chat/completions",
                    json={"model": "moonshot-v1-8k", "max_tokens": 200,
                          "messages": [{"role": "user", "content": prompt}]},
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=aiohttp.ClientTimeout(total=10),
                )
                data = await r.json()
                text = data["choices"][0]["message"]["content"]
                return {**_parse_json(text), "model": "kimi"}
        except Exception as e:
            logger.debug("Kimi error: %s", e)
            return {"error": str(e), "model": "kimi"}

    async def _ask_openrouter(self, prompt: str) -> dict:
        key = os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            return {"error": "no_key", "model": "openrouter"}
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                r = await s.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json={"model": "google/gemini-flash-1.5", "max_tokens": 200,
                          "messages": [{"role": "user", "content": prompt}]},
                    headers={"Authorization": f"Bearer {key}",
                             "HTTP-Referer": "https://centina-quant.local"},
                    timeout=aiohttp.ClientTimeout(total=10),
                )
                data = await r.json()
                text = data["choices"][0]["message"]["content"]
                return {**_parse_json(text), "model": "openrouter"}
        except Exception as e:
            logger.debug("OpenRouter error: %s", e)
            return {"error": str(e), "model": "openrouter"}

    async def _ask_gemini(self, prompt: str) -> dict:
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            return {"error": "no_key", "model": "gemini"}
        try:
            import aiohttp
            url = (
                "https://generativelanguage.googleapis.com/v1beta/"
                f"models/gemini-2.0-flash:generateContent?key={key}"
            )
            body = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 200, "temperature": 0.2},
            }
            async with aiohttp.ClientSession() as s:
                r = await s.post(url, json=body, timeout=aiohttp.ClientTimeout(total=12))
                data = await r.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return {**_parse_json(text), "model": "gemini"}
        except Exception as e:
            logger.debug("Gemini error: %s", e)
            return {"error": str(e), "model": "gemini"}

    async def _ask_openai(self, prompt: str) -> dict:
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            return {"error": "no_key", "model": "openai"}
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                r = await s.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4o-mini", "max_tokens": 200,
                          "messages": [{"role": "user", "content": prompt}]},
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=aiohttp.ClientTimeout(total=12),
                )
                data = await r.json()
                text = data["choices"][0]["message"]["content"]
                return {**_parse_json(text), "model": "openai"}
        except Exception as e:
            logger.debug("OpenAI error: %s", e)
            return {"error": str(e), "model": "openai"}

    # ── Consensus ─────────────────────────────────────────────────────

    def _compute_consensus(self, responses: list) -> dict:
        valid = [
            r for r in responses
            if isinstance(r, dict) and "verdict" in r and "error" not in r
        ]
        if not valid:
            return self._empty_consensus()

        vote_totals: dict[str, float] = {}
        confidences: list[float] = []
        reasonings: list[str] = []

        for r in valid:
            v = r.get("verdict", "PASS").upper()
            w = self._weights.get(r.get("model", ""), 1.0)
            vote_totals[v] = vote_totals.get(v, 0.0) + w
            confidences.append(float(r.get("confidence", 50)))
            model = r.get("model", "?")
            reasoning = str(r.get("reasoning", ""))[:60]
            reasonings.append(f"{model}: {reasoning}")

        best_verdict = max(vote_totals, key=vote_totals.get)
        total_votes  = sum(vote_totals.values())
        agreement    = vote_totals[best_verdict] / total_votes if total_votes > 0 else 0
        avg_conf     = sum(confidences) / len(confidences) if confidences else 50

        if best_verdict in ("PRIME", "BUY") and agreement > 0.75:
            score_bonus = 20 if best_verdict == "PRIME" else 10
        elif best_verdict in ("PRIME", "BUY") and agreement > 0.5:
            score_bonus = 5
        elif best_verdict == "PASS":
            score_bonus = -15
        elif best_verdict == "RISKY":
            score_bonus = -20
        elif agreement < 0.5:
            score_bonus = -10
        else:
            score_bonus = 0

        council_text = (
            f"{len(valid)}/{len(responses)} modèles : {best_verdict} "
            f"({agreement*100:.0f}% accord, conf {avg_conf:.0f}%)"
        )
        logger.info("LLM Council: %s", council_text)

        return {
            "verdict":             best_verdict,
            "confidence":          round(avg_conf, 1),
            "agreement_pct":       round(agreement * 100, 1),
            "score_bonus":         score_bonus,
            "models_voted":        len(valid),
            "reasoning_per_model": reasonings,
            "council_text":        council_text,
        }

    def _empty_consensus(self) -> dict:
        return {
            "verdict": "UNCERTAIN", "confidence": 0,
            "agreement_pct": 0, "score_bonus": 0,
            "models_voted": 0, "reasoning_per_model": [],
            "council_text": "Conseil LLM non disponible (aucune clé API)",
        }

    def update_accuracy(self, model: str, was_correct: bool) -> None:
        """Appelé après chaque trade pour pondérer les modèles."""
        if model not in self._accuracy:
            return
        self._accuracy[model].append(1 if was_correct else 0)
        recent = self._accuracy[model][-20:]
        self._weights[model] = max(0.3, sum(recent) / len(recent) + 0.3)


def _parse_json(text: str) -> dict:
    """Extrait le premier objet JSON valide du texte."""
    text = text.strip()
    
    # Nettoyage des éventuels blocs markdown générés par les LLMs
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Find first { ... }
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
            
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Erreur de parsing JSON de la réponse LLM : %s", e)
        return {"error": "invalid_json", "verdict": "UNCERTAIN"}
