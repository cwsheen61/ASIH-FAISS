#!/usr/bin/env python3
"""Test which Telegram channels exist and are accessible."""
import asyncio
from telethon import TelegramClient
from telethon.errors import UsernameNotOccupiedError, UsernameInvalidError
import config

async def test_channels():
    """Test a list of potential political/hate content channels."""
    session_file = config.DATA_DIR / "telegram_session"

    client = TelegramClient(
        str(session_file),
        config.TELEGRAM_API_ID,
        config.TELEGRAM_API_HASH
    )

    await client.start(phone=config.TELEGRAM_PHONE)
    print("✓ Connected to Telegram\n")

    # Channels to test - political, conspiracy, hate content
    test_channels = [
        # Political meme channels
        "conservative_memes",
        "patriotmemes",
        "trumpmemes2024",
        "bidenmemes",
        "liberaltears",
        "redalertpolitics",
        "rightwingnews",
        "maga_memes",
        "conservativememes",

        # Conspiracy/Alt-Right
        "qanon_storm",
        "thegreatawakeningchannel",
        "conspiracymemes",
        "redpilled",
        "stolenelection",
        "theredpill",
        "deepstate",

        # Anti-immigrant/Nationalist
        "borderpatrol",
        "buildthewall",
        "americafirst",
        "stoptheinvasion",
        "whitehouse",

        # Anti-politician/specific
        "aocmemes",
        "sorosmemes",
        "antiblm",
        "backtheblue",

        # General political (likely to have hateful content)
        "politicalmemesofficial",
        "worldpolitics",
        "politicalnews",
        "conservativenews",
        "liberalnews",
    ]

    valid_channels = []

    print("Testing channels...")
    print("=" * 70)

    for username in test_channels:
        try:
            entity = await client.get_entity(username)
            print(f"✓ @{username} - {entity.title}")
            valid_channels.append(username)
            await asyncio.sleep(1)  # Rate limiting
        except (UsernameNotOccupiedError, UsernameInvalidError):
            print(f"✗ @{username} - Does not exist")
        except Exception as e:
            print(f"? @{username} - Error: {e}")

    await client.disconnect()

    print("\n" + "=" * 70)
    print(f"FOUND {len(valid_channels)} VALID CHANNELS:")
    print("=" * 70)
    for channel in valid_channels:
        print(f"  @{channel}")

    return valid_channels

if __name__ == "__main__":
    valid = asyncio.run(test_channels())
