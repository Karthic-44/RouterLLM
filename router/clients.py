import time

from openai import OpenAI


def make_client(cfg):
    return OpenAI(
        base_url=cfg["base_url"],
        api_key=cfg.get("api_key") or "none",
        timeout=300,
    )


def ask(client, cfg, prompt, max_tokens):
    last = None
    for attempt in range(3):
        try:
            t0 = time.time()
            res = client.chat.completions.create(
                model=cfg["name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            secs = time.time() - t0
            pt = res.usage.prompt_tokens if res.usage else 0
            ct = res.usage.completion_tokens if res.usage else 0
            cost = (pt * cfg["price_in"] + ct * cfg["price_out"]) / 1_000_000
            return {
                "text": res.choices[0].message.content or "",
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "seconds": secs,
                "cost": cost,
            }
        except Exception as e:
            last = e
            print("request failed, trying again:", e)
            time.sleep(2 * (attempt + 1))
    raise last
