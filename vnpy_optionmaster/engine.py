from typing import Dict, List, Set, Optional
from copy import copy
from collections import defaultdict

from vnpy.trader.object import (
    LogData, ContractData, TickData,
    OrderData, TradeData, PositionData,
    SubscribeRequest, OrderRequest, CancelRequest
)
from vnpy.event import Event, EventEngine
from vnpy.trader.engine import BaseEngine, MainEngine
from vnpy.trader.event import (
    EVENT_TRADE, EVENT_TICK, EVENT_CONTRACT,
    EVENT_TIMER, EVENT_ORDER
)
from vnpy.trader.constant import (
    Product, Offset, Direction, OrderType, Exchange, Status
)
from vnpy.trader.converter import OffsetConverter, PositionHolding
from vnpy.trader.utility import extract_vt_symbol, round_to, save_json, load_json

from .base import (
    APP_NAME,
    EVENT_OPTION_NEW_PORTFOLIO,
    EVENT_OPTION_ALGO_PRICING,
    EVENT_OPTION_ALGO_TRADING,
    EVENT_OPTION_ALGO_STATUS,
    EVENT_OPTION_ALGO_LOG,
    EVENT_OPTION_RISK_NOTICE,
    InstrumentData, PortfolioData, OptionData,
    get_underlying_prefix
)
try:
    from .pricing import black_76_cython as black_76
    from .pricing import binomial_tree_cython as binomial_tree
    from .pricing import black_scholes_cython as black_scholes
except ImportError:
    from .pricing import (
        black_76, binomial_tree, black_scholes
    )
    print("Faile to import cython option pricing model, please rebuild with cython in cmd.")
from .algo import ElectronicEyeAlgo


PRICING_MODELS: dict = {
    "Black-76 欧式期货期权": black_76,
    "Black-Scholes 欧式股票期权": black_scholes,
    "二叉树 美式期货期权": binomial_tree
}


class OptionEngine(BaseEngine):
    """"""

    setting_filename: str = "option_master_setting.json"
    data_filename: str = "option_master_data.json"

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """"""
        super().__init__(main_engine, event_engine, APP_NAME)

        self.portfolios: Dict[str, PortfolioData] = {}
        self.instruments: Dict[str, InstrumentData] = {}
        self.active_portfolios: Dict[str, PortfolioData] = {}

        self.timer_count: int = 0
        self.timer_trigger: int = 60

        self.hedge_engine: OptionHedgeEngine = OptionHedgeEngine(self)
        self.algo_engine: OptionAlgoEngine = OptionAlgoEngine(self)
        self.risk_engine: OptionRiskEngine = OptionRiskEngine(self)

        self.setting: dict = {}

        self.load_setting()
        self.register_event()

    def close(self) -> None:
        """"""
        self.save_setting()
        self.save_data()

    def load_setting(self) -> None:
        """"""
        self.setting = load_json(self.setting_filename)

    def save_setting(self) -> None:
        """
        Save underlying adjustment.
        """
        save_json(self.setting_filename, self.setting)

    def load_data(self) -> None:
        """"""
        data: dict = load_json(self.data_filename)

        for portfolio in self.active_portfolios.values():
            portfolio_name: str = portfolio.name

            # Load underlying adjustment from setting
            chain_adjustments: dict = data.get("chain_adjustments", {})
            chain_adjustment_data: dict = chain_adjustments.get(portfolio_name, {})

            if chain_adjustment_data:
                for chain in portfolio.chains.values():
                    if not chain.use_synthetic:
                        chain.underlying_adjustment = chain_adjustment_data.get(
                            chain.chain_symbol, 0
                        )

            # Load pricing impv from setting
            pricing_impvs: dict = data.get("pricing_impvs", {})
            pricing_impv_data: dict = pricing_impvs.get(portfolio_name, {})

            if pricing_impv_data:
                for chain in portfolio.chains.values():
                    for index in chain.indexes:
                        key: str = f"{chain.chain_symbol}_{index}"
                        pricing_impv = pricing_impv_data.get(key, 0)

                        if pricing_impv:
                            call: OptionData = chain.calls[index]
                            call.pricing_impv = pricing_impv

                            put: OptionData = chain.puts[index]
                            put.pricing_impv = pricing_impv

    def save_data(self) -> None:
        """"""
        chain_adjustments: dict = {}
        pricing_impvs: dict = {}

        for portfolio in self.active_portfolios.values():
            chain_adjustment_data: dict = {}
            pricing_impv_data: dict = {}
            for chain in portfolio.chains.values():
                chain_adjustment_data[chain.chain_symbol] = chain.underlying_adjustment

                for call in chain.calls.values():
                    key: str = f"{chain.chain_symbol}_{call.chain_index}"
                    pricing_impv_data[key] = call.pricing_impv

            chain_adjustments[portfolio.name] = chain_adjustment_data
            pricing_impvs[portfolio.name] = pricing_impv_data

        data: dict = {
            "chain_adjustments": chain_adjustments,
            "pricing_impvs": pricing_impvs
        }

        save_json(self.data_filename, data)

    def register_event(self) -> None:
        """"""
        self.event_engine.register(EVENT_TICK, self.process_tick_event)
        self.event_engine.register(EVENT_CONTRACT, self.process_contract_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)
        self.event_engine.register(EVENT_TIMER, self.process_timer_event)

    def process_tick_event(self, event: Event) -> None:
        """"""
        tick: TickData = event.data

        instrument: Optional[InstrumentData] = self.instruments.get(tick.vt_symbol, None)
        if not instrument:
            return

        portfolio: PortfolioData = instrument.portfolio
        if not portfolio:
            return

        portfolio.update_tick(tick)

    def process_trade_event(self, event: Event) -> None:
        """"""
        trade: TradeData = event.data

        instrument: Optional[InstrumentData] = self.instruments.get(trade.vt_symbol, None)
        if not instrument:
            return

        portfolio: PortfolioData = instrument.portfolio
        if not portfolio:
            return

        portfolio.update_trade(trade)

    def process_contract_event(self, event: Event) -> None:
        """"""
        contract: ContractData = event.data

        if contract.product == Product.OPTION:
            exchange_name: str = contract.exchange.value
            portfolio_name: str = f"{contract.option_portfolio}.{exchange_name}"

            portfolio: PortfolioData = self.get_portfolio(portfolio_name)
            portfolio.add_option(contract)

    def process_timer_event(self, event: Event) -> None:
        """"""
        self.timer_count += 1
        if self.timer_count < self.timer_trigger:
            return
        self.timer_count = 0

        for portfolio in self.active_portfolios.values():
            portfolio.calculate_atm_price()

    def get_portfolio(self, portfolio_name: str) -> PortfolioData:
        """"""
        portfolio: Optional[PositionData] = self.portfolios.get(portfolio_name, None)
        if not portfolio:
            portfolio = PortfolioData(portfolio_name, self.event_engine)
            self.portfolios[portfolio_name] = portfolio

            event: Event = Event(EVENT_OPTION_NEW_PORTFOLIO, portfolio_name)
            self.event_engine.put(event)

        return portfolio

    def subscribe_data(self, vt_symbol: str) -> None:
        """"""
        contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)
        req: SubscribeRequest = SubscribeRequest(contract.symbol, contract.exchange)
        self.main_engine.subscribe(req, contract.gateway_name)

    def update_portfolio_setting(
        self,
        portfolio_name: str,
        model_name: str,
        interest_rate: float,
        chain_underlying_map: Dict[str, str],
        precision: int = 0
    ) -> None:
        """"""
        portfolio: PortfolioData = self.get_portfolio(portfolio_name)

        for chain_symbol, underlying_symbol in chain_underlying_map.items():
            if "LOCAL" in underlying_symbol:
                symbol, exchange = extract_vt_symbol(underlying_symbol)
                contract: ContractData = ContractData(
                    symbol=symbol,
                    exchange=exchange,
                    name="",
                    product=Product.INDEX,
                    size=0,
                    pricetick=0,
                    gateway_name=APP_NAME
                )
            else:
                contract: Optional[ContractData] = self.main_engine.get_contract(underlying_symbol)
            portfolio.set_chain_underlying(chain_symbol, contract)

        portfolio.set_interest_rate(interest_rate)

        pricing_model = PRICING_MODELS[model_name]
        portfolio.set_pricing_model(pricing_model)
        portfolio.set_precision(precision)

        portfolio_settings: dict = self.setting.setdefault("portfolio_settings", {})
        portfolio_settings[portfolio_name] = {
            "model_name": model_name,
            "interest_rate": interest_rate,
            "chain_underlying_map": chain_underlying_map,
            "precision": precision
        }
        self.save_setting()

    def get_portfolio_setting(self, portfolio_name: str) -> Dict:
        """"""
        portfolio_settings: dict = self.setting.setdefault("portfolio_settings", {})
        return portfolio_settings.get(portfolio_name, {})

    def init_portfolio(self, portfolio_name: str) -> bool:
        """"""
        # Add to active dict
        if portfolio_name in self.active_portfolios:
            return False
        portfolio: PortfolioData = self.get_portfolio(portfolio_name)
        self.active_portfolios[portfolio_name] = portfolio

        # Subscribe market data
        for underlying in portfolio.underlyings.values():
            if underlying.exchange == Exchange.LOCAL:
                continue

            self.instruments[underlying.vt_symbol] = underlying
            self.subscribe_data(underlying.vt_symbol)

        for option in portfolio.options.values():
            # Ignore options with no underlying set
            if not option.underlying:
                continue

            self.instruments[option.vt_symbol] = option
            self.subscribe_data(option.vt_symbol)

        # Update position volume
        for instrument in self.instruments.values():
            contract: ContractData = self.main_engine.get_contract(instrument.vt_symbol)
            converter: OffsetConverter = self.main_engine.get_converter(contract.gateway_name)
            holding: PositionHolding = converter.get_position_holding(instrument.vt_symbol)

            if holding:
                instrument.update_holding(holding)

        portfolio.calculate_pos_greeks()

        # Load chain adjustment and pricing impv data
        self.load_data()

        return True

    def get_portfolio_names(self) -> List[str]:
        """"""
        return list(self.portfolios.keys())

    def get_underlying_symbols(self, portfolio_name: str) -> List[str]:
        """"""
        underlying_prefix: str = get_underlying_prefix(portfolio_name)
        underlying_symbols: list = []

        contracts: List[ContractData] = self.main_engine.get_all_contracts()
        for contract in contracts:
            if contract.product == Product.OPTION:
                continue

            if (
                underlying_prefix
                and contract.symbol.startswith(underlying_prefix)
            ):
                underlying_symbols.append(contract.vt_symbol)

        underlying_symbols.sort()

        return underlying_symbols

    def get_instrument(self, vt_symbol: str) -> InstrumentData:
        """"""
        instrument: InstrumentData = self.instruments[vt_symbol]
        return instrument

    def set_timer_trigger(self, timer_trigger: int) -> None:
        """"""
        self.timer_trigger = timer_trigger


class OptionHedgeEngine:
    """"""

    def __init__(self, option_engine: OptionEngine) -> None:
        """"""
        self.option_engine: OptionEngine = option_engine
        self.main_engine: MainEngine = option_engine.main_engine
        self.event_engine: EventEngine = option_engine.event_engine

        # Hedging parameters
        self.portfolio_name: str = ""
        self.vt_symbol: str = ""
        self.timer_trigger: int = 5
        self.delta_target: int = 0
        self.delta_range: int = 0
        self.hedge_payup: int = 1

        self.active: bool = False
        self.active_orderids: Set[str] = set()
        self.timer_count: int = 0

        self.register_event()

    def register_event(self) -> None:
        """"""
        self.event_engine.register(EVENT_ORDER, self.process_order_event)
        self.event_engine.register(EVENT_TIMER, self.process_timer_event)

    def process_order_event(self, event: Event) -> None:
        """"""
        order: OrderData = event.data

        if order.vt_orderid not in self.active_orderids:
            return

        if not order.is_active():
            self.active_orderids.remove(order.vt_orderid)

    def process_timer_event(self, event: Event) -> None:
        """"""
        if not self.active:
            return

        self.timer_count += 1
        if self.timer_count < self.timer_trigger:
            return
        self.timer_count = 0

        self.run()

    def start(
        self,
        portfolio_name: str,
        vt_symbol: str,
        timer_trigger: int,
        delta_target: int,
        delta_range: int,
        hedge_payup: int
    ) -> None:
        """"""
        if self.active:
            return

        self.portfolio_name = portfolio_name
        self.vt_symbol = vt_symbol
        self.timer_trigger = timer_trigger
        self.delta_target = delta_target
        self.delta_range = delta_range
        self.hedge_payup = hedge_payup

        self.active = True

    def stop(self) -> None:
        """"""
        if not self.active:
            return

        self.active = False
        self.timer_count = 0

    def run(self) -> None:
        """"""
        if not self.check_order_finished():
            self.cancel_all()
            return

        delta_max = self.delta_target + self.delta_range
        delta_min = self.delta_target - self.delta_range

        # Do nothing if portfolio delta is in the allowed range
        portfolio: PortfolioData = self.option_engine.get_portfolio(self.portfolio_name)
        if delta_min <= portfolio.pos_delta <= delta_max:
            return

        # Calculate volume of contract to hedge
        delta_to_hedge = self.delta_target - portfolio.pos_delta
        instrument: InstrumentData = self.option_engine.get_instrument(self.vt_symbol)
        hedge_volume = delta_to_hedge / instrument.theo_delta

        # Send hedge orders
        tick: Optional[TickData] = self.main_engine.get_tick(self.vt_symbol)
        contract: Optional[ContractData] = self.main_engine.get_contract(self.vt_symbol)
        converter: OffsetConverter = self.main_engine.get_converter(contract.gateway_name)
        holding: PositionHolding = converter.get_position_holding(self.vt_symbol)

        # Check if hedge volume meets contract minimum trading volume
        if abs(hedge_volume) < contract.min_volume:
            return

        if hedge_volume > 0:
            price: float = tick.ask_price_1 + contract.pricetick * self.hedge_payup
            direction: Direction = Direction.LONG

            if holding:
                available = holding.short_pos - holding.short_pos_frozen
            else:
                available = 0
        else:
            price: float = tick.bid_price_1 - contract.pricetick * self.hedge_payup
            direction: Direction = Direction.SHORT

            if holding:
                available = holding.long_pos - holding.long_pos_frozen
            else:
                available = 0

        order_volume = abs(hedge_volume)

        req: OrderRequest = OrderRequest(
            symbol=contract.symbol,
            exchange=contract.exchange,
            direction=direction,
            type=OrderType.LIMIT,
            volume=order_volume,
            price=round_to(price, contract.pricetick),
            reference=f"{APP_NAME}_DeltaHedging"
        )

        # Close positon if opposite available is enough
        if available > order_volume:
            req.offset = Offset.CLOSE
            vt_orderid: str = self.main_engine.send_order(req, contract.gateway_name)
            self.active_orderids.add(vt_orderid)
        # Open position if no oppsite available
        elif not available:
            req.offset = Offset.OPEN
            vt_orderid: str = self.main_engine.send_order(req, contract.gateway_name)
            self.active_orderids.add(vt_orderid)
        # Else close all opposite available and open left volume
        else:
            close_req: OrderRequest = copy(req)
            close_req.offset = Offset.CLOSE
            close_req.volume = available
            close_orderid: str = self.main_engine.send_order(close_req, contract.gateway_name)
            self.active_orderids.add(close_orderid)

            open_req: OrderRequest = copy(req)
            open_req.offset = Offset.OPEN
            open_req.volume = order_volume - available
            open_orderid: str = self.main_engine.send_order(open_req, contract.gateway_name)
            self.active_orderids.add(open_orderid)

    def check_order_finished(self) -> bool:
        """"""
        if self.active_orderids:
            return False
        else:
            return True

    def cancel_all(self) -> None:
        """"""
        for vt_orderid in self.active_orderids:
            order: Optional[OrderData] = self.main_engine.get_order(vt_orderid)
            req: CancelRequest = order.create_cancel_request()
            self.main_engine.cancel_order(req, order.gateway_name)


class OptionAlgoEngine:

    def __init__(self, option_engine: OptionEngine) -> None:
        """"""
        self.option_engine: OptionEngine = option_engine
        self.main_engine: MainEngine = option_engine.main_engine
        self.event_engine: EventEngine = option_engine.event_engine

        self.algos: Dict[str, ElectronicEyeAlgo] = {}
        self.active_algos: Dict[str, ElectronicEyeAlgo] = {}

        self.underlying_algo_map: Dict[str, ElectronicEyeAlgo] = defaultdict(list)
        self.order_algo_map: Dict[str, ElectronicEyeAlgo] = {}

        self.register_event()

    def init_engine(self, portfolio_name: str) -> None:
        """"""
        if self.algos:
            return

        portfolio: PortfolioData = self.option_engine.get_portfolio(portfolio_name)

        for option in portfolio.options.values():
            algo: ElectronicEyeAlgo = ElectronicEyeAlgo(self, option)
            self.algos[option.vt_symbol] = algo

    def register_event(self) -> None:
        """"""
        self.event_engine.register(EVENT_ORDER, self.process_order_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)
        self.event_engine.register(EVENT_TIMER, self.process_timer_event)

    def process_underlying_tick_event(self, event: Event) -> None:
        """"""
        tick: TickData = event.data

        for algo in self.underlying_algo_map[tick.vt_symbol]:
            algo.on_underlying_tick(tick)

    def process_option_tick_event(self, event: Event) -> None:
        """"""
        tick: TickData = event.data

        algo: ElectronicEyeAlgo = self.algos[tick.vt_symbol]
        algo.on_option_tick(tick)

    def process_order_event(self, event: Event) -> None:
        """"""
        order: OrderData = event.data
        algo: Optional[ElectronicEyeAlgo] = self.order_algo_map.get(order.vt_orderid, None)

        if algo:
            algo.on_order(order)

    def process_trade_event(self, event: Event) -> None:
        """"""
        trade: TradeData = event.data
        algo: Optional[ElectronicEyeAlgo] = self.order_algo_map.get(trade.vt_orderid, None)

        if algo:
            algo.on_trade(trade)

    def process_timer_event(self, event: Event) -> None:
        """"""
        for algo in self.active_algos.values():
            algo.on_timer()

    def start_algo_pricing(self, vt_symbol: str, params: dict) -> None:
        """"""
        algo: ElectronicEyeAlgo = self.algos[vt_symbol]

        result: bool = algo.start_pricing(params)
        if not result:
            return

        self.underlying_algo_map[algo.underlying.vt_symbol].append(algo)
        self.active_algos[vt_symbol] = algo

        self.event_engine.register(
            EVENT_TICK + algo.option.vt_symbol,
            self.process_option_tick_event
        )
        self.event_engine.register(
            EVENT_TICK + algo.underlying.vt_symbol,
            self.process_underlying_tick_event
        )

    def stop_algo_pricing(self, vt_symbol: str) -> None:
        """"""
        algo: ElectronicEyeAlgo = self.algos[vt_symbol]

        result: bool = algo.stop_pricing()
        if not result:
            return

        self.event_engine.unregister(
            EVENT_TICK + vt_symbol,
            self.process_option_tick_event
        )

        buf: list = self.underlying_algo_map[algo.underlying.vt_symbol]
        buf.remove(algo)

        self.active_algos.pop(vt_symbol)

        if not buf:
            self.event_engine.unregister(
                EVENT_TICK + algo.underlying.vt_symbol,
                self.process_underlying_tick_event
            )

    def start_algo_trading(self, vt_symbol: str, params: dict) -> None:
        """"""
        algo: ElectronicEyeAlgo = self.algos[vt_symbol]
        algo.start_trading(params)

    def stop_algo_trading(self, vt_symbol: str) -> None:
        """"""
        algo: ElectronicEyeAlgo = self.algos[vt_symbol]
        algo.stop_trading()

    def send_order(
        self,
        algo: ElectronicEyeAlgo,
        vt_symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: int
    ) -> str:
        """"""
        contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)

        req: OrderRequest = OrderRequest(
            contract.symbol,
            contract.exchange,
            direction,
            OrderType.LIMIT,
            volume,
            price,
            offset,
            reference=f"{APP_NAME}_ElectronicEye"
        )

        vt_orderid: str = self.main_engine.send_order(req, contract.gateway_name)
        self.order_algo_map[vt_orderid] = algo

        return vt_orderid

    def cancel_order(self, vt_orderid: str) -> None:
        """"""
        order: Optional[OrderData] = self.main_engine.get_order(vt_orderid)
        req: CancelRequest = order.create_cancel_request()
        self.main_engine.cancel_order(req, order.gateway_name)

    def write_algo_log(self, algo: ElectronicEyeAlgo, msg: str) -> None:
        """"""
        msg: str = f"[{algo.vt_symbol}] {msg}"
        log: LogData = LogData(APP_NAME, msg)
        event: Event = Event(EVENT_OPTION_ALGO_LOG, log)
        self.event_engine.put(event)

    def put_algo_pricing_event(self, algo: ElectronicEyeAlgo) -> None:
        """"""
        event: Event = Event(EVENT_OPTION_ALGO_PRICING, algo)
        self.event_engine.put(event)

    def put_algo_trading_event(self, algo: ElectronicEyeAlgo) -> None:
        """"""
        event: Event = Event(EVENT_OPTION_ALGO_TRADING, algo)
        self.event_engine.put(event)

    def put_algo_status_event(self, algo: ElectronicEyeAlgo) -> None:
        """"""
        event: Event = Event(EVENT_OPTION_ALGO_STATUS, algo)
        self.event_engine.put(event)


class OptionRiskEngine:
    """期权风控引擎"""

    def __init__(self, option_engine: OptionEngine) -> None:
        """"""
        self.option_engine: OptionEngine = option_engine
        self.event_engine: EventEngine = option_engine.event_engine

        self.instruments: Dict[str, InstrumentData] = option_engine.instruments

        # 成交持仓比风控
        self.trade_volume: int = 0
        self.net_pos: int = 0

        # 委托撤单比风控
        self.all_orderids: set[str] = set()
        self.cancel_orderids: set[str] = set()

        # 定时运行参数
        self.timer_count: int = 0
        self.timer_trigger: int = 10

        self.register_event()

    def register_event(self) -> None:
        """"""
        self.event_engine.register(EVENT_ORDER, self.process_order_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)
        self.event_engine.register(EVENT_TIMER, self.process_timer_event)

    def process_order_event(self, event: Event) -> None:
        """"""
        order: OrderData = event.data
        self.all_orderids.add(order.vt_orderid)

        if order.status == Status.CANCELLED:
            self.cancel_orderids.add(order.vt_orderid)

    def process_trade_event(self, event: Event) -> None:
        """"""
        trade: TradeData = event.data

        self.trade_volume += trade.volume

    def process_timer_event(self, event: Event) -> None:
        """"""
        self.timer_count += 1
        if self.timer_count < self.timer_trigger:
            return
        self.timer_count = 0

        self.net_pos = 0
        for instrument in self.instruments.values():
            self.net_pos += instrument.net_pos

        self.put_event()

    def put_event(self) -> None:
        """推送事件"""
        order_count: int = len(self.all_orderids)
        cancel_count: int = len(self.cancel_orderids)

        if self.net_pos:
            trade_position_ratio: float = self.trade_volume / abs(self.net_pos)
        else:
            trade_position_ratio: float = 9999

        if order_count:
            cancel_order_ratio: float = cancel_count / order_count
        else:
            cancel_order_ratio: float = 0

        data: dict = {
            "trade_volume": self.trade_volume,
            "net_pos": self.net_pos,
            "order_count": len(self.all_orderids),
            "cancel_count": len(self.cancel_orderids),
            "trade_position_ratio": trade_position_ratio,
            "cancel_order_ratio": cancel_order_ratio
        }
        self.event_engine.put(Event(EVENT_OPTION_RISK_NOTICE, data))
