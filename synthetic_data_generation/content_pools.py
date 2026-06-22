"""
content_pools.py

Sampling helpers for per-conversation content diversity.
Imported by generate_conversations.py and generate_probe_conversations.py.
"""

import math
import random
import datetime

MERCHANTS = [
    "Amazon", "Walmart", "Target", "Best Buy", "Apple Store", "Netflix",
    "Spotify", "Uber", "Uber Eats", "DoorDash", "Lyft", "Shell", "Chevron",
    "Starbucks", "Costco", "Home Depot", "Steam", "PlayStation Store",
    "Adobe", "Microsoft", "Google Play", "Airbnb", "Expedia", "Delta Air Lines",
    "CVS Pharmacy", "Walgreens", "Nike", "Etsy", "eBay", "PayPal",
    "Cash App", "Venmo", "Zelle transfer", "Wayfair", "Instacart", "GrubHub",
    "AT&T", "Verizon", "Comcast", "Planet Fitness", "Peloton", "Ticketmaster",
    "an unfamiliar online store", "a gas station I don't recognize",
    "some subscription I don't remember", "a foreign merchant", "Temu",
    "Shein", "AliExpress", "a hotel in another city",
]

# Held-out merchants reserved ONLY for the generalization test set (Phase 5)
MERCHANTS_HELDOUT = [
    "Chipotle", "REI", "Sephora", "Square checkout", "Roblox",
    "a parking garage", "a medical clinic", "an unknown ATM withdrawal",
]

CARD_TYPES = ["Visa", "Mastercard", "Amex", "debit card", "credit card"]
CHANNELS   = ["online", "in-store", "recurring subscription", "ATM", "phone order"]


def sample_amount(rng: random.Random) -> str:
    lo, hi = math.log(3), math.log(3000)
    val = math.exp(rng.uniform(lo, hi))
    if rng.random() < 0.15:
        val = round(val / 10) * 10
    return f"${val:,.2f}"


def sample_date(rng: random.Random, ref_date=None, heldout: bool = False) -> str:
    ref = ref_date or datetime.date.today()
    # Heldout uses a disjoint window (181–365 days ago) vs training (1–180 days ago)
    lo, hi = (181, 365) if heldout else (1, 180)
    d = ref - datetime.timedelta(days=rng.randint(lo, hi))
    month = d.strftime("%B")
    day = d.day  # int, no leading zero
    return f"{month} {day}"  # e.g. "March 4"


def sample_target_length(rng: random.Random, communication_style: str) -> int:
    """Return a soft per-conversation word-count target drawn from a persona-matched distribution."""
    if communication_style == "terse":
        return max(3, int(rng.triangular(3, 14, 6)))
    elif communication_style == "direct":
        return max(6, int(rng.triangular(8, 28, 15)))
    elif communication_style == "indirect":
        return max(12, int(rng.triangular(18, 60, 30)))
    elif communication_style == "verbose":
        base = int(rng.triangular(35, 72, 50))
        if rng.random() < 0.2:
            base = int(rng.triangular(72, 92, 80))
        return base
    return 15  # fallback for unknown styles


def sample_content(rng: random.Random, heldout: bool = False) -> dict:
    pool = MERCHANTS_HELDOUT if heldout else MERCHANTS
    return {
        "merchant_name":    rng.choice(pool),
        "amount":           sample_amount(rng),
        "transaction_date": sample_date(rng, heldout=heldout),
        "card_type":        rng.choice(CARD_TYPES),
        "channel":          rng.choice(CHANNELS),
    }
