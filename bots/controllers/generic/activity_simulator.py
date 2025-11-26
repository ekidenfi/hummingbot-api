"""
Activity Simulator - generates round-trip trades with zero net position.
Uses actual exchange position as source of truth.
"""
import random
from decimal import Decimal
from typing import List, Optional

from pydantic import Field

from hummingbot.core.data_type.common import MarketDict, PositionAction, PositionMode, PriceType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.data_types import ConnectorPair
from hummingbot.strategy_v2.executors.order_executor.data_types import ExecutionStrategy, OrderExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo


class ActivitySimulatorConfig(ControllerConfigBase):
    controller_type: str = "generic"
    controller_name: str = "activity_simulator"
    candles_config: List[CandlesConfig] = []

    connector_name: str = "ekiden_perpetual"
    trading_pair: str = "ENA-USDC"
    leverage: int = 1
    position_mode: PositionMode = PositionMode.ONEWAY

    use_orderbook_prices: bool = Field(default=True, json_schema_extra={"is_updatable": True})
    max_orderbook_depth: int = Field(default=10, json_schema_extra={"is_updatable": True})
    spread_bps: Decimal = Field(default=Decimal("10"), json_schema_extra={"is_updatable": True})

    min_order_size_quote: Decimal = Field(default=Decimal("10"), json_schema_extra={"is_updatable": True})
    max_order_size_quote: Decimal = Field(default=Decimal("50"), json_schema_extra={"is_updatable": True})

    order_interval_seconds: float = Field(default=2.0, json_schema_extra={"is_updatable": True})
    stale_order_seconds: float = Field(default=5.0, json_schema_extra={"is_updatable": True})

    def update_markets(self, markets: MarketDict) -> MarketDict:
        return markets.add_or_update(self.connector_name, self.trading_pair)


class ActivitySimulator(ControllerBase):

    def __init__(self, config: ActivitySimulatorConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self._last_order_time = 0.0
        self._last_open_side: Optional[TradeType] = None
        self._completed_roundtrips = 0
        self._replaced_orders = 0
        self.initialize_rate_sources()

    def initialize_rate_sources(self):
        self.market_data_provider.initialize_rate_sources([
            ConnectorPair(connector_name=self.config.connector_name, trading_pair=self.config.trading_pair)
        ])

    def active_executors(self) -> List[ExecutorInfo]:
        return [e for e in self.executors_info if e.is_active]

    def get_exchange_position(self) -> Decimal:
        try:
            connector = self.market_data_provider.get_connector(self.config.connector_name)
            if hasattr(connector, 'get_position'):
                pos = connector.get_position(self.config.trading_pair)
                if pos is not None:
                    return pos.amount
            if hasattr(connector, '_perpetual_trading'):
                for key, pos in connector._perpetual_trading.account_positions.items():
                    if self.config.trading_pair in key:
                        return pos.amount
        except Exception:
            pass
        return Decimal("0")

    def generate_random_size_quote(self) -> Decimal:
        return Decimal(str(round(random.uniform(
            float(self.config.min_order_size_quote),
            float(self.config.max_order_size_quote)
        ), 2)))

    def get_crossing_price(self, side: TradeType) -> Optional[Decimal]:
        try:
            order_book = self.market_data_provider.get_order_book(
                self.config.connector_name, self.config.trading_pair
            )
            entries = list(order_book.ask_entries() if side == TradeType.BUY else order_book.bid_entries())
            if entries:
                depth = min(self.config.max_orderbook_depth, len(entries))
                return Decimal(str(random.choice(entries[:depth]).price))
        except Exception:
            pass
        return None

    def calculate_order_price(self, side: TradeType) -> Decimal:
        if self.config.use_orderbook_prices:
            price = self.get_crossing_price(side)
            if price:
                return price

        mid_price = self.market_data_provider.get_price_by_type(
            self.config.connector_name, self.config.trading_pair, PriceType.MidPrice
        )
        spread = self.config.spread_bps / Decimal("10000")
        return mid_price * (Decimal("1") + spread) if side == TradeType.BUY else mid_price * (Decimal("1") - spread)

    def check_stale_executors(self) -> List[ExecutorAction]:
        actions = []
        current_time = self.market_data_provider.time()
        for executor in self.active_executors():
            if current_time - executor.timestamp >= self.config.stale_order_seconds:
                actions.append(StopExecutorAction(
                    controller_id=self.config.id, executor_id=executor.id, keep_position=True
                ))
                self._replaced_orders += 1
        return actions

    def should_place_order(self) -> bool:
        current_time = self.market_data_provider.time()
        return (current_time - self._last_order_time >= self.config.order_interval_seconds
                and len(self.active_executors()) == 0)

    def determine_executor_actions(self) -> List[ExecutorAction]:
        stale_actions = self.check_stale_executors()
        if stale_actions:
            return stale_actions

        if not self.should_place_order():
            return []

        mid_price = self.market_data_provider.get_price_by_type(
            self.config.connector_name, self.config.trading_pair, PriceType.MidPrice
        )
        if not mid_price or mid_price <= 0:
            return []

        position = self.get_exchange_position()

        if position == Decimal("0"):
            if self._last_open_side is None:
                side = random.choice([TradeType.BUY, TradeType.SELL])
            else:
                side = TradeType.SELL if self._last_open_side == TradeType.BUY else TradeType.BUY
                self._completed_roundtrips += 1

            size_quote = self.generate_random_size_quote()
            price = self.calculate_order_price(side)
            amount = size_quote / price
            position_action = PositionAction.OPEN
            self._last_open_side = side
        else:
            side = TradeType.BUY if position < 0 else TradeType.SELL
            amount = abs(position)
            price = self.calculate_order_price(side)
            position_action = PositionAction.CLOSE

        self._last_order_time = self.market_data_provider.time()

        return [CreateExecutorAction(
            controller_id=self.config.id,
            executor_config=OrderExecutorConfig(
                timestamp=self._last_order_time,
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                side=side,
                amount=amount,
                price=price,
                execution_strategy=ExecutionStrategy.LIMIT,
                position_action=position_action,
                leverage=self.config.leverage,
                level_id=f"sim_{position_action.name.lower()}_{self._last_order_time}",
            )
        )]

    async def update_processed_data(self):
        pass

    def to_format_status(self) -> List[str]:
        mid_price = self.market_data_provider.get_price_by_type(
            self.config.connector_name, self.config.trading_pair, PriceType.MidPrice
        )
        position = self.get_exchange_position()
        state = "FLAT" if position == Decimal("0") else ("LONG" if position > 0 else "SHORT")
        last = self._last_open_side.name if self._last_open_side else "-"
        next_side = "SELL" if self._last_open_side == TradeType.BUY else "BUY" if self._last_open_side else "?"

        return [
            "┌" + "─" * 80 + "┐",
            f"│ Activity Simulator: {self.config.trading_pair:<56} │",
            "├" + "─" * 80 + "┤",
            f"│ Mid: {mid_price:.6f}  │  Position: {position:+.4f}  │  State: {state:<20}│",
            f"│ Last: {last:<4} Next: {next_side:<4} │ Trips: {self._completed_roundtrips:<4} │ Replaced: {self._replaced_orders:<4} │ Active: {len(self.active_executors()):<2}│",
            "└" + "─" * 80 + "┘",
        ]
