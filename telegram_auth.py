"""Telegram authentication helper."""
import asyncio
from telethon import TelegramClient
import config

async def authenticate():
    """Authenticate with Telegram using the verification code."""
    session_file = config.DATA_DIR / "telegram_session"

    client = TelegramClient(
        str(session_file),
        config.TELEGRAM_API_ID,
        config.TELEGRAM_API_HASH
    )

    # Connect and authenticate with the code
    await client.connect()

    if not await client.is_user_authorized():
        await client.send_code_request(config.TELEGRAM_PHONE)
        print("Enter the code sent to your phone:")
        code = "44569"  # The code provided by user

        try:
            await client.sign_in(config.TELEGRAM_PHONE, code)
            print("✓ Successfully authenticated!")
        except Exception as e:
            print(f"Error: {e}")
            return False
    else:
        print("✓ Already authenticated")

    await client.disconnect()
    return True

if __name__ == "__main__":
    asyncio.run(authenticate())
