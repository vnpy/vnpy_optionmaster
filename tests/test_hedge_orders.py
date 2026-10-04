from collections.abc import Callable
from datetime import datetime
from typing import cast

from vnpy.trader.constant import Direction, Exchange, Offset, OrderType, Product
from vnpy.trader.converter import PositionHolding
from vnpy.trader.object import ContractData, OrderRequest, TickData

from vnpy_optionmaster.base import APP_NAME
from vnpy_optionmaster.engine import OptionEngine, OptionHedgeEngine


_PORTFOLIO_NAME: str = "IF_PORTFOLIO"
_DELTA_TARGET: int = 4
_DELTA_RANGE: int = 1
_THEO_DELTA: float = 2.0
_HEDGE_PAYUP: int = 2
_ASK_PRICE: float = 10.0
_BID_PRICE: float = 9.5
_PRICE_TICK: float = 0.5
_GATEWAY_NAME: str = "TEST"


class _EventEngine:
    def __init__(self) -> None:
        self.handlers: list[tuple[str, Callable[..., None]]] = []

    def register(self, event_type: str, handler: Callable[..., None]) -> None:
        self.handlers.append((event_type, handler))


class _HoldingBook:
    def __init__(self, holding: PositionHolding) -> None:
        self.holding: PositionHolding = holding

    def get_position_holding(self, vt_symbol: str) -> PositionHolding:
        return self.holding


class _MainEngine:
    def __init__(self, contract: ContractData, tick: TickData, holding: PositionHolding) -> None:
        self.contract: ContractData = contract
        self.tick: TickData = tick
        self.book: _HoldingBook = _HoldingBook(holding)
        self.calls: list[tuple[OrderRequest, str]] = []
        self._seq: int = 0

    def get_tick(self, vt_symbol: str) -> TickData | None:
        if vt_symbol == self.contract.vt_symbol:
            return self.tick
        return None

    def get_contract(self, vt_symbol: str) -> ContractData | None:
        if vt_symbol == self.contract.vt_symbol:
            return self.contract
        return None

    def get_converter(self, gateway_name: str) -> _HoldingBook | None:
        if gateway_name == self.contract.gateway_name:
            return self.book
        return None

    def send_order(self, req: OrderRequest, gateway_name: str) -> str:
        self._seq += 1
        self.calls.append((req, gateway_name))
        return f"{gateway_name}.{self._seq}"


class _Portfolio:
    def __init__(self, pos_delta: float) -> None:
        self.pos_delta: float = pos_delta


class _Instrument:
    def __init__(self, theo_delta: float) -> None:
        self.theo_delta: float = theo_delta


class _OptionEngine:
    def __init__(
        self,
        main_engine: _MainEngine,
        portfolio: _Portfolio,
        instrument: _Instrument,
    ) -> None:
        self.main_engine: _MainEngine = main_engine
        self.event_engine: _EventEngine = _EventEngine()
        self._portfolio: _Portfolio = portfolio
        self._instrument: _Instrument = instrument

    def get_portfolio(self, portfolio_name: str) -> _Portfolio:
        assert portfolio_name == _PORTFOLIO_NAME
        return self._portfolio

    def get_instrument(self, vt_symbol: str) -> _Instrument:
        assert vt_symbol == self.main_engine.contract.vt_symbol
        return self._instrument


class _HedgeResult:
    def __init__(
        self,
        hedge: OptionHedgeEngine,
        calls: list[tuple[OrderRequest, str]],
        tick: TickData,
        contract: ContractData,
    ) -> None:
        self.hedge: OptionHedgeEngine = hedge
        self.calls: list[tuple[OrderRequest, str]] = calls
        self.tick: TickData = tick
        self.contract: ContractData = contract


def _order_volume(pos_delta: float, theo_delta: float = _THEO_DELTA) -> float:
    return abs((_DELTA_TARGET - pos_delta) / theo_delta)


def _long_hedge_price() -> float:
    return _ASK_PRICE + _PRICE_TICK * _HEDGE_PAYUP


def _short_hedge_price() -> float:
    return _BID_PRICE - _PRICE_TICK * _HEDGE_PAYUP


def _contract(min_volume: float) -> ContractData:
    return ContractData(
        gateway_name=_GATEWAY_NAME,
        symbol="IF2412",
        exchange=Exchange.CFFEX,
        name="IF2412",
        product=Product.FUTURES,
        size=1,
        pricetick=_PRICE_TICK,
        min_volume=min_volume,
    )


def _tick(contract: ContractData) -> TickData:
    return TickData(
        gateway_name=contract.gateway_name,
        symbol=contract.symbol,
        exchange=contract.exchange,
        datetime=datetime(2026, 10, 4, 9, 30),
        ask_price_1=_ASK_PRICE,
        ask_volume_1=10,
        bid_price_1=_BID_PRICE,
        bid_volume_1=10,
    )


def _holding(
    contract: ContractData,
    long_pos: float,
    long_pos_frozen: float,
    short_pos: float,
    short_pos_frozen: float,
) -> PositionHolding:
    holding: PositionHolding = PositionHolding(contract)
    holding.long_pos = long_pos
    holding.long_pos_frozen = long_pos_frozen
    holding.short_pos = short_pos
    holding.short_pos_frozen = short_pos_frozen
    return holding


def _run_hedge(
    pos_delta: float,
    long_pos: float,
    long_pos_frozen: float,
    short_pos: float,
    short_pos_frozen: float,
    theo_delta: float = _THEO_DELTA,
    min_volume: float = 1,
) -> _HedgeResult:
    contract: ContractData = _contract(min_volume)
    tick: TickData = _tick(contract)
    holding: PositionHolding = _holding(
        contract,
        long_pos,
        long_pos_frozen,
        short_pos,
        short_pos_frozen,
    )
    main_engine: _MainEngine = _MainEngine(contract, tick, holding)
    option_engine: _OptionEngine = _OptionEngine(
        main_engine,
        _Portfolio(pos_delta),
        _Instrument(theo_delta),
    )
    hedge: OptionHedgeEngine = OptionHedgeEngine(cast(OptionEngine, option_engine))
    hedge.portfolio_name = _PORTFOLIO_NAME
    hedge.vt_symbol = contract.vt_symbol
    hedge.delta_target = _DELTA_TARGET
    hedge.delta_range = _DELTA_RANGE
    hedge.hedge_payup = _HEDGE_PAYUP
    hedge.run()
    return _HedgeResult(hedge, main_engine.calls, tick, contract)


def _assert_single(
    result: _HedgeResult,
    direction: Direction,
    offset: Offset,
    price: float,
    volume: float,
) -> None:
    assert len(result.calls) == 1
    req: OrderRequest
    gateway_name: str
    req, gateway_name = result.calls[0]
    assert gateway_name == result.contract.gateway_name
    assert req.symbol == result.contract.symbol
    assert req.exchange == result.contract.exchange
    assert req.direction == direction
    assert req.offset == offset
    assert req.type == OrderType.LIMIT
    assert req.volume == volume
    assert req.price == price
    assert req.reference == f"{APP_NAME}_DeltaHedging"


def _assert_close_then_open(
    result: _HedgeResult,
    direction: Direction,
    price: float,
    available: float,
    order_volume: float,
) -> None:
    assert len(result.calls) == 2
    close_req: OrderRequest
    open_req: OrderRequest
    close_gateway: str
    open_gateway: str
    close_req, close_gateway = result.calls[0]
    open_req, open_gateway = result.calls[1]
    assert close_gateway == result.contract.gateway_name
    assert open_gateway == result.contract.gateway_name
    assert close_req.direction == direction
    assert open_req.direction == direction
    assert close_req.type == OrderType.LIMIT
    assert open_req.type == OrderType.LIMIT
    assert close_req.price == price
    assert open_req.price == price
    assert close_req.offset == Offset.CLOSE
    assert open_req.offset == Offset.OPEN
    assert close_req.volume == available
    assert open_req.volume == order_volume - available
    assert close_req.volume > 0
    assert open_req.volume > 0


class TestHedgeOrders:
    def test_close_when_opposite_available_exceeds_hedge_volume(self) -> None:
        pos_delta: float = -6.0
        volume: float = _order_volume(pos_delta)
        short_pos: float = 9.0
        short_pos_frozen: float = 2.0
        available: float = short_pos - short_pos_frozen
        assert available > volume
        long_result: _HedgeResult = _run_hedge(
            pos_delta,
            long_pos=0.0,
            long_pos_frozen=0.0,
            short_pos=short_pos,
            short_pos_frozen=short_pos_frozen,
        )
        _assert_single(
            long_result,
            Direction.LONG,
            Offset.CLOSE,
            _long_hedge_price(),
            volume,
        )

        pos_delta = 14.0
        volume = _order_volume(pos_delta)
        long_pos: float = 9.0
        long_pos_frozen: float = 2.0
        available = long_pos - long_pos_frozen
        assert available > volume
        short_result: _HedgeResult = _run_hedge(
            pos_delta,
            long_pos=long_pos,
            long_pos_frozen=long_pos_frozen,
            short_pos=0.0,
            short_pos_frozen=0.0,
        )
        _assert_single(
            short_result,
            Direction.SHORT,
            Offset.CLOSE,
            _short_hedge_price(),
            volume,
        )

    def test_open_when_no_opposite_position(self) -> None:
        pos_delta: float = -6.0
        volume: float = _order_volume(pos_delta)
        short_pos: float = 0.0
        short_pos_frozen: float = 0.0
        assert short_pos - short_pos_frozen == 0
        long_result: _HedgeResult = _run_hedge(
            pos_delta,
            long_pos=9.0,
            long_pos_frozen=0.0,
            short_pos=short_pos,
            short_pos_frozen=short_pos_frozen,
        )
        _assert_single(
            long_result,
            Direction.LONG,
            Offset.OPEN,
            _long_hedge_price(),
            volume,
        )

        pos_delta = 14.0
        volume = _order_volume(pos_delta)
        long_pos: float = 0.0
        long_pos_frozen: float = 0.0
        assert long_pos - long_pos_frozen == 0
        short_result: _HedgeResult = _run_hedge(
            pos_delta,
            long_pos=long_pos,
            long_pos_frozen=long_pos_frozen,
            short_pos=9.0,
            short_pos_frozen=0.0,
        )
        _assert_single(
            short_result,
            Direction.SHORT,
            Offset.OPEN,
            _short_hedge_price(),
            volume,
        )

    def test_close_then_open_when_opposite_available_is_smaller(self) -> None:
        pos_delta: float = -6.0
        volume: float = _order_volume(pos_delta)
        short_pos: float = 5.0
        short_pos_frozen: float = 3.0
        available: float = short_pos - short_pos_frozen
        assert available > 0
        assert available < volume
        long_result: _HedgeResult = _run_hedge(
            pos_delta,
            long_pos=0.0,
            long_pos_frozen=0.0,
            short_pos=short_pos,
            short_pos_frozen=short_pos_frozen,
        )
        _assert_close_then_open(
            long_result,
            Direction.LONG,
            _long_hedge_price(),
            available,
            volume,
        )

        pos_delta = 14.0
        volume = _order_volume(pos_delta)
        long_pos: float = 5.0
        long_pos_frozen: float = 3.0
        available = long_pos - long_pos_frozen
        assert available > 0
        assert available < volume
        short_result: _HedgeResult = _run_hedge(
            pos_delta,
            long_pos=long_pos,
            long_pos_frozen=long_pos_frozen,
            short_pos=0.0,
            short_pos_frozen=0.0,
        )
        _assert_close_then_open(
            short_result,
            Direction.SHORT,
            _short_hedge_price(),
            available,
            volume,
        )

    def test_no_order_when_hedge_volume_below_min_volume(self) -> None:
        pos_delta: float = 0.0
        theo_delta: float = 10.0
        min_volume: float = 1.0
        hedge_volume: float = _order_volume(pos_delta, theo_delta)
        delta_min: float = _DELTA_TARGET - _DELTA_RANGE
        delta_max: float = _DELTA_TARGET + _DELTA_RANGE
        assert hedge_volume < min_volume
        assert pos_delta < delta_min or pos_delta > delta_max
        result: _HedgeResult = _run_hedge(
            pos_delta,
            long_pos=0.0,
            long_pos_frozen=0.0,
            short_pos=0.0,
            short_pos_frozen=0.0,
            theo_delta=theo_delta,
            min_volume=min_volume,
        )
        assert result.calls == []
