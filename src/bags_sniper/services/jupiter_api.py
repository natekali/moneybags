"""
Jupiter Aggregator API client for fallback trading.
Used when Bags.fm API is unavailable or returns errors.

Jupiter API Docs: https://station.jup.ag/docs/apis/swap-api
"""

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

import httpx
import structlog
import base58
import base64

from bags_sniper.core.config import Settings

logger = structlog.get_logger()

# Jupiter API endpoints
JUPITER_QUOTE_API = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_API = "https://quote-api.jup.ag/v6/swap"

# SOL mint address
SOL_MINT = "So11111111111111111111111111111111111111112"


@dataclass
class JupiterResponse:
    """Standardized Jupiter API response."""

    success: bool
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    status_code: int = 0


class JupiterAPIClient:
    """
    Async client for Jupiter Aggregator API.
    Used as fallback when Bags.fm API returns errors.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Optional[httpx.AsyncClient] = None

    async def connect(self):
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={
                "Content-Type": "application/json",
            },
        )
        logger.info("jupiter_api_connected")

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "JupiterAPIClient":
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    def _get_keypair(self):
        """Get Solana keypair from settings."""
        from solders.keypair import Keypair
        import json

        private_key_str = self.settings.wallet_private_key.get_secret_value().strip()

        # Try JSON array format first
        if private_key_str.startswith('['):
            try:
                key_array = json.loads(private_key_str)
                if isinstance(key_array, list) and len(key_array) == 64:
                    return Keypair.from_bytes(bytes(key_array))
            except json.JSONDecodeError:
                pass

        # Try base58
        try:
            private_key_bytes = base58.b58decode(private_key_str)
            if len(private_key_bytes) == 64:
                return Keypair.from_bytes(private_key_bytes)
            elif len(private_key_bytes) == 32:
                return Keypair.from_seed(private_key_bytes)
        except Exception:
            pass

        # Try base64
        try:
            private_key_bytes = base64.b64decode(private_key_str)
            if len(private_key_bytes) == 64:
                return Keypair.from_bytes(private_key_bytes)
            elif len(private_key_bytes) == 32:
                return Keypair.from_seed(private_key_bytes)
        except Exception:
            pass

        raise ValueError("Could not decode private key for Jupiter")

    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 500,
    ) -> JupiterResponse:
        """
        Get a swap quote from Jupiter.

        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount in smallest unit
            slippage_bps: Slippage tolerance in basis points
        """
        if not self._client:
            await self.connect()

        try:
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": slippage_bps,
            }

            response = await self._client.get(JUPITER_QUOTE_API, params=params)

            if response.status_code >= 400:
                return JupiterResponse(
                    success=False,
                    error=f"Jupiter quote error: {response.text}",
                    status_code=response.status_code,
                )

            return JupiterResponse(
                success=True,
                data=response.json(),
                status_code=response.status_code,
            )

        except Exception as e:
            logger.error("jupiter_quote_error", error=str(e))
            return JupiterResponse(success=False, error=str(e))

    async def create_swap_transaction(
        self,
        quote_response: dict,
        user_public_key: str,
    ) -> JupiterResponse:
        """
        Create a swap transaction from a Jupiter quote.

        Args:
            quote_response: The quote response from get_quote
            user_public_key: User's wallet public key
        """
        if not self._client:
            await self.connect()

        try:
            payload = {
                "quoteResponse": quote_response,
                "userPublicKey": user_public_key,
                "wrapAndUnwrapSol": True,
                "dynamicComputeUnitLimit": True,
                "prioritizationFeeLamports": "auto",
            }

            response = await self._client.post(JUPITER_SWAP_API, json=payload)

            if response.status_code >= 400:
                return JupiterResponse(
                    success=False,
                    error=f"Jupiter swap error: {response.text}",
                    status_code=response.status_code,
                )

            return JupiterResponse(
                success=True,
                data=response.json(),
                status_code=response.status_code,
            )

        except Exception as e:
            logger.error("jupiter_swap_error", error=str(e))
            return JupiterResponse(success=False, error=str(e))

    async def sign_and_submit(
        self,
        swap_transaction: str,
        rpc_client,
    ) -> JupiterResponse:
        """
        Sign and submit a Jupiter swap transaction.

        Args:
            swap_transaction: Base64 encoded transaction from Jupiter
            rpc_client: Solana RPC client for submission
        """
        from solders.transaction import VersionedTransaction
        from solders.message import to_bytes_versioned

        try:
            keypair = self._get_keypair()

            # Jupiter returns base64 encoded transactions
            tx_bytes = base64.b64decode(swap_transaction)

            logger.debug(
                "jupiter_tx_decoding",
                tx_length=len(tx_bytes),
            )

            # Parse and sign the transaction
            tx = VersionedTransaction.from_bytes(tx_bytes)
            message_bytes = to_bytes_versioned(tx.message)
            signature = keypair.sign_message(message_bytes)

            # Create signed transaction
            signed_tx = VersionedTransaction.populate(tx.message, [signature])
            signed_tx_bytes = bytes(signed_tx)

            # Submit via RPC
            response = await rpc_client.send_transaction(signed_tx_bytes)

            if response.success:
                logger.info(
                    "jupiter_tx_submitted",
                    signature=response.result,
                )
                return JupiterResponse(
                    success=True,
                    data={"signature": response.result},
                )
            else:
                return JupiterResponse(
                    success=False,
                    error=response.error or "Transaction submission failed",
                )

        except Exception as e:
            logger.error("jupiter_sign_submit_error", error=str(e))
            return JupiterResponse(success=False, error=str(e))

    async def buy_token(
        self,
        mint_address: str,
        amount_sol: Decimal,
        slippage_bps: int = 500,
        rpc_client=None,
    ) -> JupiterResponse:
        """
        Execute a buy order through Jupiter.

        Args:
            mint_address: Token mint address
            amount_sol: Amount in SOL to spend
            slippage_bps: Slippage tolerance in basis points
            rpc_client: Solana RPC client for submission
        """
        if mint_address == SOL_MINT:
            return JupiterResponse(
                success=False,
                error="Cannot buy SOL with SOL",
            )

        logger.info(
            "jupiter_executing_buy",
            mint=mint_address[:8],
            amount_sol=str(amount_sol),
        )

        try:
            keypair = self._get_keypair()
            user_pubkey = str(keypair.pubkey())

            # Convert SOL to lamports
            amount_lamports = int(amount_sol * Decimal("1e9"))

            # Step 1: Get quote
            quote_response = await self.get_quote(
                input_mint=SOL_MINT,
                output_mint=mint_address,
                amount=amount_lamports,
                slippage_bps=slippage_bps,
            )

            if not quote_response.success:
                return quote_response

            quote_data = quote_response.data
            logger.info(
                "jupiter_quote_received",
                mint=mint_address[:8],
                in_amount=quote_data.get("inAmount"),
                out_amount=quote_data.get("outAmount"),
            )

            # Step 2: Create swap transaction
            swap_response = await self.create_swap_transaction(
                quote_response=quote_data,
                user_public_key=user_pubkey,
            )

            if not swap_response.success:
                return swap_response

            swap_tx = swap_response.data.get("swapTransaction")
            if not swap_tx:
                return JupiterResponse(
                    success=False,
                    error="No swap transaction in Jupiter response",
                )

            # Step 3: Sign and submit
            if rpc_client is None:
                from bags_sniper.services.solana_rpc import SolanaRPCClient
                rpc_client = SolanaRPCClient(self.settings)
                await rpc_client.connect()
                should_close = True
            else:
                should_close = False

            try:
                submit_response = await self.sign_and_submit(swap_tx, rpc_client)

                if submit_response.success:
                    return JupiterResponse(
                        success=True,
                        data={
                            "signature": submit_response.data.get("signature"),
                            "tokens_received": quote_data.get("outAmount"),
                            "price": amount_sol / Decimal(quote_data.get("outAmount", 1)),
                        },
                    )
                else:
                    return submit_response
            finally:
                if should_close:
                    await rpc_client.close()

        except Exception as e:
            logger.error("jupiter_buy_error", error=str(e), mint=mint_address[:8])
            return JupiterResponse(success=False, error=str(e))

    async def sell_token(
        self,
        mint_address: str,
        amount_tokens: int,
        slippage_bps: int = 1000,
        rpc_client=None,
    ) -> JupiterResponse:
        """
        Execute a sell order through Jupiter.

        Args:
            mint_address: Token mint address
            amount_tokens: Amount of tokens to sell
            slippage_bps: Slippage tolerance in basis points
            rpc_client: Solana RPC client for submission
        """
        logger.info(
            "jupiter_executing_sell",
            mint=mint_address[:8],
            amount_tokens=amount_tokens,
        )

        try:
            keypair = self._get_keypair()
            user_pubkey = str(keypair.pubkey())

            # Step 1: Get quote (Token -> SOL)
            quote_response = await self.get_quote(
                input_mint=mint_address,
                output_mint=SOL_MINT,
                amount=amount_tokens,
                slippage_bps=slippage_bps,
            )

            if not quote_response.success:
                return quote_response

            quote_data = quote_response.data
            logger.info(
                "jupiter_sell_quote_received",
                mint=mint_address[:8],
                in_amount=quote_data.get("inAmount"),
                out_amount=quote_data.get("outAmount"),
            )

            # Step 2: Create swap transaction
            swap_response = await self.create_swap_transaction(
                quote_response=quote_data,
                user_public_key=user_pubkey,
            )

            if not swap_response.success:
                return swap_response

            swap_tx = swap_response.data.get("swapTransaction")
            if not swap_tx:
                return JupiterResponse(
                    success=False,
                    error="No swap transaction in Jupiter response",
                )

            # Step 3: Sign and submit
            if rpc_client is None:
                from bags_sniper.services.solana_rpc import SolanaRPCClient
                rpc_client = SolanaRPCClient(self.settings)
                await rpc_client.connect()
                should_close = True
            else:
                should_close = False

            try:
                submit_response = await self.sign_and_submit(swap_tx, rpc_client)

                if submit_response.success:
                    sol_received = Decimal(quote_data.get("outAmount", 0)) / Decimal("1e9")
                    return JupiterResponse(
                        success=True,
                        data={
                            "signature": submit_response.data.get("signature"),
                            "sol_received": sol_received,
                            "tokens_sold": amount_tokens,
                        },
                    )
                else:
                    return submit_response
            finally:
                if should_close:
                    await rpc_client.close()

        except Exception as e:
            logger.error("jupiter_sell_error", error=str(e), mint=mint_address[:8])
            return JupiterResponse(success=False, error=str(e))
