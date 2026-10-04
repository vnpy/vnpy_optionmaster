from collections.abc import Callable
from datetime import datetime
from typing import cast

from vnpy.trader.constant import Direction, Exchange, Offset, OrderType, Product
from vnpy.trader.object import ContractData, OrderRequest, TickData

from vnpy_optionmaster.algo import ElectronicEyeAlgo
from vnpy_optionmaster.base import APP_NAME, OptionData
from vnpy_optionmaster.engine import OptionAlgoEngine, OptionEngine


_ASK_PRICE: float = 1.25
_ALGO_BID_PRICE: float = 1.5
_GATEWAY_NAME: str = "TEST"
_SYMBOL: str = "10004829"


class _EventEngine:
    def __init__(self) -> None:
        self.handlers: list[tuple[str, Callable[..., None]]] = []
        self.events: list[object] = []

    def register(self, event_type: str, handler: Callable[..., None]) -> None:
        self.handlers.append((event_type, handler))

    def put(self, event: object) -> None:
        self.events.append(event)


class _UnderlyingView:
    def __init__(self) -> None:
        self.vt_symbol: str = "510050.SSE"


class _OptionView:
    def __init__(self, vt_symbol: str, short_pos: int, net_pos: int, tick: TickData | None) -> None:
        self.vt_symbol: str = vt_symbol
        self.pricetick: float = 0.0001
        self.underlying: _UnderlyingView = _UnderlyingView()
        self.short_pos: int = short_pos
        self.net_pos: int = net_pos
        self.tick: TickData | None = tick


class _MainEngine:
    def __init__(self, contract: ContractData) -> None:
        self.contract: ContractData = contract
        self.calls: list[tuple[OrderRequest, str]] = []
        self.orderids: list[str] = []
        self._seq: int = 0

    def get_contract(self, vt_symbol: str) -> ContractData | None:
        if vt_symbol == self.contract.vt_symbol:
            return self.contract
        return None

    def send_order(self, req: OrderRequest, gateway_name: str) -> str:
        self._seq += 1
        vt_orderid: str = f"{gateway_name}.{self._seq}"
        self.calls.append((req, gateway_name))
        self.orderids.append(vt_orderid)
        return vt_orderid


class _OptionEngine:
    def __init__(self, main_engine: _MainEngine) -> None:
        self.main_engine: _MainEngine = main_engine
        self.event_engine: _EventEngine = _EventEngine()


def _contract() -> ContractData:
    return ContractData(
        gateway_name=_GATEWAY_NAME,
        symbol=_SYMBOL,
        exchange=Exchange.SSE,
        name="50ETF购12月",
        product=Product.OPTION,
        size=10000,
        pricetick=0.0001,
        min_volume=1,
    )


def _tick(ask_volume: float) -> TickData:
    return TickData(
        gateway_name=_GATEWAY_NAME,
        symbol=_SYMBOL,
        exchange=Exchange.SSE,
        datetime=datetime(2026, 10, 4, 9, 30),
        ask_price_1=_ASK_PRICE,
        ask_volume_1=ask_volume,
        bid_price_1=1.0,
        bid_volume_1=10,
    )


def _make_algo(
    short_pos: int,
    net_pos: int,
    tick: TickData | None,
) -> tuple[ElectronicEyeAlgo, _MainEngine, OptionAlgoEngine]:
    contract: ContractData = _contract()
    main_engine: _MainEngine = _MainEngine(contract)
    option_engine: _OptionEngine = _OptionEngine(main_engine)
    algo_engine: OptionAlgoEngine = OptionAlgoEngine(cast(OptionEngine, option_engine))
    option: _OptionView = _OptionView(contract.vt_symbol, short_pos, net_pos, tick)
    algo: ElectronicEyeAlgo = ElectronicEyeAlgo(algo_engine, cast(OptionData, option))
    return algo, main_engine, algo_engine


def _assert_long_order(req: OrderRequest, offset: Offset, price: float, volume: float) -> None:
    assert req.direction == Direction.LONG
    assert req.offset == offset
    assert req.type == OrderType.LIMIT
    assert req.price == price
    assert req.volume == volume
    assert req.reference == f"{APP_NAME}_ElectronicEye"


class TestAlgoOrders:
    def test_send_order_limit_reference_and_order_map(self) -> None:
        algo: ElectronicEyeAlgo
        main_engine: _MainEngine
        algo_engine: OptionAlgoEngine
        algo, main_engine, algo_engine = _make_algo(short_pos=0, net_pos=0, tick=None)
        direction: Direction = Direction.SHORT
        offset: Offset = Offset.CLOSE
        price: float = 2.5
        volume: int = 4

        vt_orderid: str = algo_engine.send_order(
            algo,
            algo.vt_symbol,
            direction,
            offset,
            price,
            volume,
        )

        assert vt_orderid != ""
        assert len(main_engine.calls) == 1
        req: OrderRequest
        gateway_name: str
        req, gateway_name = main_engine.calls[0]
        assert gateway_name == main_engine.contract.gateway_name
        assert req.symbol == main_engine.contract.symbol
        assert req.exchange == main_engine.contract.exchange
        assert req.direction == direction
        assert req.offset == offset
        assert req.type == OrderType.LIMIT
        assert req.price == price
        assert req.volume == volume
        assert "ElectronicEye" in req.reference
        assert req.reference == f"{APP_NAME}_ElectronicEye"
        assert algo_engine.order_algo_map[vt_orderid] is algo
        assert main_engine.orderids == [vt_orderid]

    def test_send_long_opens_when_no_short(self) -> None:
        price: float = 1.8
        volume: int = 6
        algo: ElectronicEyeAlgo
        main_engine: _MainEngine
        algo_engine: OptionAlgoEngine
        algo, main_engine, algo_engine = _make_algo(short_pos=0, net_pos=0, tick=None)
        assert algo.option.short_pos == 0

        algo.send_long(price, volume)

        assert len(main_engine.calls) == 1
        _assert_long_order(main_engine.calls[0][0], Offset.OPEN, price, volume)
        assert algo_engine.order_algo_map[main_engine.orderids[0]] is algo

    def test_send_long_closes_when_short_is_enough(self) -> None:
        price: float = 1.8
        volume: int = 3
        short_pos: int = 8
        algo: ElectronicEyeAlgo
        main_engine: _MainEngine
        algo_engine: OptionAlgoEngine
        algo, main_engine, algo_engine = _make_algo(short_pos=short_pos, net_pos=-short_pos, tick=None)
        assert algo.option.short_pos >= volume

        algo.send_long(price, volume)

        assert len(main_engine.calls) == 1
        _assert_long_order(main_engine.calls[0][0], Offset.CLOSE, price, volume)
        assert algo_engine.order_algo_map[main_engine.orderids[0]] is algo

    def test_send_long_closes_then_opens_when_short_is_partial(self) -> None:
        price: float = 1.8
        volume: int = 8
        short_pos: int = 3
        algo: ElectronicEyeAlgo
        main_engine: _MainEngine
        algo_engine: OptionAlgoEngine
        algo, main_engine, algo_engine = _make_algo(short_pos=short_pos, net_pos=-short_pos, tick=None)
        assert 0 < algo.option.short_pos < volume

        algo.send_long(price, volume)

        assert len(main_engine.calls) == 2
        close_req: OrderRequest = main_engine.calls[0][0]
        open_req: OrderRequest = main_engine.calls[1][0]
        _assert_long_order(close_req, Offset.CLOSE, price, short_pos)
        _assert_long_order(open_req, Offset.OPEN, price, volume - short_pos)
        assert close_req.volume > 0
        assert open_req.volume > 0
        assert [algo_engine.order_algo_map[vt_orderid] for vt_orderid in main_engine.orderids] == [algo, algo]

    def test_snipe_long_volume_within_left_ask_and_max_size(self) -> None:
        cases: tuple[tuple[int, float, int], ...] = (
            (4, 9.0, 8),
            (10, 3.0, 8),
            (10, 9.0, 2),
        )
        max_pos: int
        ask_volume: float
        max_order_size: int
        for max_pos, ask_volume, max_order_size in cases:
            tick: TickData = _tick(ask_volume)
            algo: ElectronicEyeAlgo
            main_engine: _MainEngine
            algo_engine: OptionAlgoEngine
            algo, main_engine, algo_engine = _make_algo(short_pos=0, net_pos=0, tick=tick)
            algo.algo_bid_price = _ALGO_BID_PRICE
            algo.target_pos = 0
            algo.max_pos = max_pos
            algo.max_order_size = max_order_size
            assert tick.ask_price_1 <= algo.algo_bid_price
            assert algo.option.short_pos == 0

            algo.snipe_long()

            volume_left: int = algo.target_pos + algo.max_pos - algo.option.net_pos
            capped: float = min(volume_left, tick.ask_volume_1, algo.max_order_size)
            assert len(main_engine.calls) == 1
            req: OrderRequest = main_engine.calls[0][0]
            _assert_long_order(req, Offset.OPEN, algo.algo_bid_price, capped)
            assert req.volume <= volume_left
            assert req.volume <= tick.ask_volume_1
            assert req.volume <= algo.max_order_size
            assert algo_engine.order_algo_map[main_engine.orderids[0]] is algo
