"""Tests for storage/supported_assets.py — word-boundary asset matching."""

from storage.supported_assets import match_asset


class TestMatchAsset:
    def test_match_bitcoin(self):
        assert match_asset("Will Bitcoin reach $100K?")["coin_id"] == "bitcoin"

    def test_match_btc(self):
        assert match_asset("BTC all time high by June")["coin_id"] == "bitcoin"

    def test_match_ethereum(self):
        assert match_asset("Ethereum all time high by March")["coin_id"] == "ethereum"

    def test_match_eth(self):
        assert match_asset("Will ETH reach $5000?")["coin_id"] == "ethereum"

    def test_match_solana(self):
        assert match_asset("Will SOL reach $500?")["coin_id"] == "solana"

    def test_match_solana_full(self):
        assert match_asset("Solana price prediction")["coin_id"] == "solana"

    def test_no_match(self):
        assert match_asset("Will Netherlands win the FIFA World Cup?") is None

    def test_no_match_edgex(self):
        assert match_asset("EdgeX FDV above $500M one day after launch?") is None

    def test_false_positive_eth_in_netherlands(self):
        """'Netherlands' contains 'eth' substring but should NOT match."""
        assert match_asset("Netherlands win World Cup") is None

    def test_false_positive_sol_in_resolution(self):
        """'resolution' contains 'sol' substring but should NOT match."""
        assert match_asset("Will a specific resolution be approved?") is None

    def test_case_insensitive(self):
        assert match_asset("BITCOIN price")["coin_id"] == "bitcoin"
        assert match_asset("bitcoin price")["coin_id"] == "bitcoin"
        assert match_asset("Bitcoin Price")["coin_id"] == "bitcoin"

    def test_match_doge(self):
        assert match_asset("Will DOGE reach $1?")["coin_id"] == "dogecoin"
        assert match_asset("Dogecoin price prediction")["coin_id"] == "dogecoin"

    def test_match_xrp(self):
        assert match_asset("Will XRP reach $5?")["coin_id"] == "ripple"

    def test_match_link(self):
        assert match_asset("LINK price above $50?")["coin_id"] == "chainlink"

    def test_match_ada(self):
        assert match_asset("Cardano reach $2?")["coin_id"] == "cardano"

    def test_match_avax(self):
        assert match_asset("AVAX price prediction")["coin_id"] == "avalanche-2"

    def test_match_bnb(self):
        assert match_asset("BNB above $600?")["coin_id"] == "binancecoin"

    def test_match_ton(self):
        assert match_asset("TON price reach $10?")["coin_id"] == "the-open-network"

    def test_match_dot(self):
        assert match_asset("Polkadot reach $20?")["coin_id"] == "polkadot"

    def test_match_ltc(self):
        assert match_asset("Litecoin above $100?")["coin_id"] == "litecoin"

    def test_returns_coin_id_and_symbol(self):
        result = match_asset("BTC to $200K")
        assert result["coin_id"] == "bitcoin"
        assert result["symbol"] == "BTC"
