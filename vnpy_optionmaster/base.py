from datetime import datetime
from typing import Dict, List, Callable, Optional
from types import ModuleType
from functools import lru_cache

from vnpy.event import EventEngine
from vnpy.event.engine import Event
from vnpy.trader.event import EVENT_TICK
from vnpy.trader.object import ContractData, TickData, TradeData
from vnpy.trader.constant import Exchange, OptionType, Direction, Offset
from vnpy.trader.converter import PositionHolding
from vnpy.trader.utility import extract_vt_symbol

from .time import calculate_days_to_expiry, ANNUAL_DAYS


APP_NAME = "OptionMaster"

EVENT_OPTION_NEW_PORTFOLIO = "eOptionNewPortfolio"
EVENT_OPTION_ALGO_PRICING = "eOptionAlgoPricing"
EVENT_OPTION_ALGO_TRADING = "eOptionAlgoTrading"
EVENT_OPTION_ALGO_STATUS = "eOptionAlgoStatus"
EVENT_OPTION_ALGO_LOG = "eOptionAlgoLog"
EVENT_OPTION_RISK_NOTICE = "eOptionRiskNotice"


class InstrumentData:
    """"""

    def __init__(self, contract: ContractData) -> None:
        """"""
        self.symbol: str = contract.symbol
        self.exchange: Exchange = contract.exchange
        self.vt_symbol: str = contract.vt_symbol

        self.pricetick: float = contract.pricetick
        self.min_volume: float = contract.min_volume
        self.size: int = contract.size

        self.long_pos: int = 0
        self.short_pos: int = 0
        self.net_pos: int = 0
        self.mid_price: float = 0

        self.tick: TickData = None
        self.portfolio: PortfolioData = None

    def calculate_net_pos(self) -> None:
        """"""
        self.net_pos = self.long_pos - self.short_pos

    def update_tick(self, tick: TickData) -> None:
        """"""
        self.tick = tick
        self.mid_price = (tick.bid_price_1 + tick.ask_price_1) / 2

    def update_trade(self, trade: TradeData) -> None:
        """"""
        if trade.direction == Direction.LONG:
            if trade.offset == Offset.OPEN:
                self.long_pos += trade.volume
            else:
                self.short_pos -= trade.volume
        else:
            if trade.offset == Offset.OPEN:
                self.short_pos += trade.volume
            else:
                self.long_pos -= trade.volume
        self.calculate_net_pos()

    def update_holding(self, holding: PositionHolding) -> None:
        """"""
        self.long_pos = holding.long_pos
        self.short_pos = holding.short_pos
        self.calculate_net_pos()

    def set_portfolio(self, portfolio: "PortfolioData") -> None:
        """"""
        self.portfolio = portfolio


class OptionData(InstrumentData):
    """"""

    def __init__(self, contract: ContractData) -> None:
        """"""
        super().__init__(contract)

        # Option contract features
        self.strike_price: float = contract.option_strike
        self.chain_index: str = contract.option_index

        self.option_type: int = 0
        if contract.option_type == OptionType.CALL:
            self.option_type = 1
        else:
            self.option_type = -1

        self.option_expiry: datetime = contract.option_expiry
        self.days_to_expiry: int = calculate_days_to_expiry(
            contract.option_expiry
        )
        self.time_to_expiry: float = self.days_to_expiry / ANNUAL_DAYS

        self.interest_rate: float = 0

        # Option portfolio related
        self.underlying: UnderlyingData = None
        self.chain: ChainData = None
        self.underlying_adjustment: float = 0

        # Pricing model
        self.calculate_price: Callable = None
        self.calculate_greeks: Callable = None
        self.calculate_impv: Callable = None

        # Implied volatility
        self.bid_impv: float = 0
        self.ask_impv: float = 0
        self.mid_impv: float = 0
        self.pricing_impv: float = 0

        # Greeks related
        self.theo_delta: float = 0
        self.theo_gamma: float = 0
        self.theo_theta: float = 0
        self.theo_vega: float = 0

        self.pos_value: float = 0
        self.pos_delta: float = 0
        self.pos_gamma: float = 0
        self.pos_theta: float = 0
        self.pos_vega: float = 0

    def calculate_option_impv(self) -> None:
        """"""
        if not self.tick or not self.underlying:
            return

        underlying_price: float = self.underlying.mid_price
        if not underlying_price:
            return
        underlying_price += self.underlying_adjustment

        ask_price: float = self.tick.ask_price_1
        bid_price: float = self.tick.bid_price_1

        if ask_price and bid_price:
            mid_price: float = (ask_price + bid_price) / 2
        elif ask_price:
            mid_price = ask_price
        elif bid_price:
            mid_price = bid_price
        else:
            mid_price = 0

        self.ask_impv = self.calculate_impv(
            ask_price,
            underlying_price,
            self.strike_price,
            self.interest_rate,
            self.time_to_expiry,
            self.option_type
        )

        self.bid_impv = self.calculate_impv(
            bid_price,
            underlying_price,
            self.strike_price,
            self.interest_rate,
            self.time_to_expiry,
            self.option_type
        )

        self.mid_impv = self.calculate_impv(
            mid_price,
            underlying_price,
            self.strike_price,
            self.interest_rate,
            self.time_to_expiry,
            self.option_type
        )

    def calculate_theo_greeks(self) -> None:
        """"""
        if not self.underlying:
            return

        underlying_price: float = self.underlying.mid_price
        if not underlying_price or not self.mid_impv:
            return
        underlying_price += self.underlying_adjustment

        _, delta, gamma, theta, vega = self.calculate_greeks(
            underlying_price,
            self.strike_price,
            self.interest_rate,
            self.time_to_expiry,
            self.mid_impv,
            self.option_type
        )

        self.theo_delta = delta * self.size
        self.theo_gamma = gamma * self.size
        self.theo_theta = theta * self.size / 240
        self.theo_vega = vega * self.size / 100

    def calculate_pos_greeks(self) -> None:
        """"""
        if self.tick:
            self.pos_value = self.tick.last_price * self.size * self.net_pos

        self.pos_delta = self.theo_delta * self.net_pos
        self.pos_gamma = self.theo_gamma * self.net_pos
        self.pos_theta = self.theo_theta * self.net_pos
        self.pos_vega = self.theo_vega * self.net_pos

    def calculate_ref_price(self) -> float:
        """"""
        underlying_price: float = self.underlying.mid_price
        underlying_price += self.underlying_adjustment

        ref_price: float = self.calculate_price(
            underlying_price,
            self.strike_price,
            self.interest_rate,
            self.time_to_expiry,
            self.pricing_impv,
            self.option_type
        )

        return ref_price

    def update_tick(self, tick: TickData) -> None:
        """"""
        super().update_tick(tick)

        self.calculate_option_impv()

    def update_trade(self, trade: TradeData) -> None:
        """"""
        super().update_trade(trade)
        self.calculate_pos_greeks()

    def update_underlying_tick(self, underlying_adjustment: float) -> None:
        """"""
        self.underlying_adjustment = underlying_adjustment

        self.calculate_option_impv()
        self.calculate_theo_greeks()
        self.calculate_pos_greeks()

    def set_chain(self, chain: "ChainData") -> None:
        """"""
        self.chain = chain

    def set_underlying(self, underlying: "UnderlyingData") -> None:
        """"""
        self.underlying = underlying

    def set_interest_rate(self, interest_rate: float) -> None:
        """"""
        self.interest_rate = interest_rate

    def set_pricing_model(self, pricing_model: ModuleType) -> None:
        """"""
        self.calculate_greeks = pricing_model.calculate_greeks
        self.calculate_impv = pricing_model.calculate_impv
        self.calculate_price = pricing_model.calculate_price


class UnderlyingData(InstrumentData):
    """"""

    def __init__(self, contract: ContractData) -> None:
        """"""
        super().__init__(contract)

        self.theo_delta: float = self.size                  # 标的物理论Delta固定为1
        self.pos_delta: float = 0
        self.chains: Dict[str, ChainData] = {}

    def add_chain(self, chain: "ChainData") -> None:
        """"""
        self.chains[chain.chain_symbol] = chain

    def update_tick(self, tick: TickData) -> None:
        """"""
        super().update_tick(tick)

        for chain in self.chains.values():
            chain.update_underlying_tick()

        self.calculate_pos_greeks()

    def update_trade(self, trade: TradeData) -> None:
        """"""
        super().update_trade(trade)

        self.calculate_pos_greeks()

    def calculate_pos_greeks(self) -> None:
        """"""
        self.pos_delta = self.theo_delta * self.net_pos


class ChainData:
    """"""

    def __init__(self, chain_symbol: str, event_engine: EventEngine) -> None:
        """"""
        self.chain_symbol: str = chain_symbol
        self.event_engine: EventEngine = event_engine

        self.long_pos: int = 0
        self.short_pos: int = 0
        self.net_pos: int = 0

        self.pos_value: float = 0
        self.pos_delta: float = 0
        self.pos_gamma: float = 0
        self.pos_theta: float = 0
        self.pos_vega: float = 0

        self.underlying: UnderlyingData = None

        self.options: Dict[str, OptionData] = {}
        self.calls: Dict[str, OptionData] = {}
        self.puts: Dict[str, OptionData] = {}

        self.portfolio: PortfolioData = None

        self.indexes: List[float] = []
        self.atm_price: float = 0
        self.atm_index: str = ""
        self.underlying_adjustment: float = 0
        self.days_to_expiry: int = 0

        self.use_synthetic: bool = False

    def add_option(self, option: OptionData) -> None:
        """"""
        self.options[option.vt_symbol] = option

        if option.option_type > 0:
            self.calls[option.chain_index] = option
        else:
            self.puts[option.chain_index] = option

        option.set_chain(self)

        if option.chain_index not in self.indexes:
            self.indexes.append(option.chain_index)

            # Sort index by number if possible, otherwise by string
            try:
                float(option.chain_index)
                self.indexes.sort(key=float)
            except ValueError:
                self.indexes.sort()

        self.days_to_expiry = option.days_to_expiry

    def calculate_pos_greeks(self) -> None:
        """"""
        # Clear data
        self.long_pos = 0
        self.short_pos = 0
        self.net_pos = 0
        self.pos_value = 0
        self.pos_delta = 0
        self.pos_gamma = 0
        self.pos_theta = 0
        self.pos_vega = 0

        # Sum all value
        for option in self.options.values():
            if option.net_pos:
                self.long_pos += option.long_pos
                self.short_pos += option.short_pos
                self.pos_value += option.pos_value
                self.pos_delta += option.pos_delta
                self.pos_gamma += option.pos_gamma
                self.pos_theta += option.pos_theta
                self.pos_vega += option.pos_vega

        self.net_pos = self.long_pos - self.short_pos

    def update_tick(self, tick: TickData) -> None:
        """"""
        option: OptionData = self.options[tick.vt_symbol]
        option.update_tick(tick)

        if self.use_synthetic:
            if not self.atm_index:
                self.calculate_atm_price()

            if option.chain_index == self.atm_index:
                self.update_synthetic_price()

    def update_underlying_tick(self) -> None:
        """"""
        if not self.use_synthetic:
            self.calculate_underlying_adjustment()

        for option in self.options.values():
            option.update_underlying_tick(self.underlying_adjustment)

        self.calculate_pos_greeks()

    def update_trade(self, trade: TradeData) -> None:
        """"""
        option: OptionData = self.options[trade.vt_symbol]

        # Deduct old option pos greeks
        self.long_pos -= option.long_pos
        self.short_pos -= option.short_pos
        self.pos_value -= option.pos_value
        self.pos_delta -= option.pos_delta
        self.pos_gamma -= option.pos_gamma
        self.pos_theta -= option.pos_theta
        self.pos_vega -= option.pos_vega

        # Calculate new option pos greeks
        option.update_trade(trade)

        # Add new option pos greeks
        self.long_pos += option.long_pos
        self.short_pos += option.short_pos
        self.pos_value += option.pos_value
        self.pos_delta += option.pos_delta
        self.pos_gamma += option.pos_gamma
        self.pos_theta += option.pos_theta
        self.pos_vega += option.pos_vega

        self.net_pos = self.long_pos - self.short_pos

    def set_underlying(self, underlying: "UnderlyingData") -> None:
        """"""
        underlying.add_chain(self)
        self.underlying = underlying

        for option in self.options.values():
            option.set_underlying(underlying)

        if underlying.exchange == Exchange.LOCAL:
            self.use_synthetic = True

    def set_interest_rate(self, interest_rate: float) -> None:
        """"""
        for option in self.options.values():
            option.set_interest_rate(interest_rate)

    def set_pricing_model(self, pricing_model: ModuleType) -> None:
        """"""
        for option in self.options.values():
            option.set_pricing_model(pricing_model)

    def set_portfolio(self, portfolio: "PortfolioData") -> None:
        """"""
        for option in self.options.values():
            option.set_portfolio(portfolio)

    def calculate_atm_price(self) -> None:
        """"""
        min_diff = 0
        atm_price = 0
        atm_index: str = ""

        for index, call in self.calls.items():
            put: OptionData = self.puts[index]

            call_tick: TickData = call.tick
            if not call_tick or not call_tick.bid_price_1 or not call_tick.ask_price_1:
                continue

            put_tick: TickData = put.tick
            if not put_tick or not put_tick.bid_price_1 or not put_tick.ask_price_1:
                continue

            call_mid_price: float = (call_tick.ask_price_1 + call_tick.bid_price_1) / 2
            put_mid_price: float = (put_tick.ask_price_1 + put_tick.bid_price_1) / 2

            diff: float = abs(call_mid_price - put_mid_price)

            if not min_diff or diff < min_diff:
                min_diff: float = diff
                atm_price: float = call.strike_price
                atm_index: str = call.chain_index

        self.atm_price = atm_price
        self.atm_index = atm_index

    def calculate_underlying_adjustment(self) -> None:
        """"""
        if not self.atm_price:
            return

        atm_call: OptionData = self.calls[self.atm_index]
        atm_put: OptionData = self.puts[self.atm_index]

        call_price: float = atm_call.mid_price
        put_price: float = atm_put.mid_price

        synthetic_price: float = call_price - put_price + self.atm_price
        self.underlying_adjustment = synthetic_price - self.underlying.mid_price

    def update_synthetic_price(self) -> None:
        """"""
        call: OptionData = self.calls[self.atm_index]
        put: OptionData = self.puts[self.atm_index]

        self.underlying.mid_price = call.mid_price - put.mid_price + self.atm_price
        self.update_underlying_tick()

        # 推送合成期货的行情
        symbol, exchange = extract_vt_symbol(self.underlying.vt_symbol)

        tick: TickData = TickData(
            symbol=symbol,
            exchange=exchange,
            datetime=datetime.now(),
            last_price=self.underlying.mid_price,
            gateway_name=APP_NAME
        )
        event: Event = Event(EVENT_TICK + tick.vt_symbol, tick)
        self.event_engine.put(event)


class PortfolioData:

    def __init__(self, name: str, event_engine: EventEngine) -> None:
        """"""
        self.name: str = name
        self.event_engine: EventEngine = event_engine

        self.long_pos: int = 0
        self.short_pos: int = 0
        self.net_pos: int = 0

        self.pos_delta: float = 0
        self.pos_gamma: float = 0
        self.pos_theta: float = 0
        self.pos_vega: float = 0

        # All instrument
        self._options: Dict[str, OptionData] = {}
        self._chains: Dict[str, ChainData] = {}

        # Active instrument
        self.options: Dict[str, OptionData] = {}
        self.chains: Dict[str, ChainData] = {}
        self.underlyings: Dict[str, UnderlyingData] = {}

        # Greeks decimals precision
        self.precision: int = 0

    def calculate_pos_greeks(self) -> None:
        """"""
        self.long_pos = 0
        self.short_pos = 0
        self.net_pos = 0

        self.pos_value = 0
        self.pos_delta = 0
        self.pos_gamma = 0
        self.pos_theta = 0
        self.pos_vega = 0

        for underlying in self.underlyings.values():
            self.pos_delta += underlying.pos_delta

        for chain in self.chains.values():
            self.long_pos += chain.long_pos
            self.short_pos += chain.short_pos
            self.pos_value += chain.pos_value
            self.pos_delta += chain.pos_delta
            self.pos_gamma += chain.pos_gamma
            self.pos_theta += chain.pos_theta
            self.pos_vega += chain.pos_vega

        self.net_pos = self.long_pos - self.short_pos

    def update_tick(self, tick: TickData) -> None:
        """"""
        if tick.vt_symbol in self.options:
            option: OptionData = self.options[tick.vt_symbol]
            chain: ChainData = option.chain
            chain.update_tick(tick)
            self.calculate_pos_greeks()
        elif tick.vt_symbol in self.underlyings:
            underlying: UnderlyingData = self.underlyings[tick.vt_symbol]
            underlying.update_tick(tick)
            self.calculate_pos_greeks()

    def update_trade(self, trade: TradeData) -> None:
        """"""
        if trade.vt_symbol in self.options:
            option: OptionData = self.options[trade.vt_symbol]
            chain: ChainData = option.chain
            chain.update_trade(trade)
            self.calculate_pos_greeks()
        elif trade.vt_symbol in self.underlyings:
            underlying: UnderlyingData = self.underlyings[trade.vt_symbol]
            underlying.update_trade(trade)
            self.calculate_pos_greeks()

    def set_interest_rate(self, interest_rate: float) -> None:
        """"""
        for chain in self.chains.values():
            chain.set_interest_rate(interest_rate)

    def set_pricing_model(self, pricing_model: ModuleType) -> None:
        """"""
        for chain in self.chains.values():
            chain.set_pricing_model(pricing_model)

    def set_precision(self, precision: int) -> None:
        """"""
        self.precision = precision

    def set_chain_underlying(self, chain_symbol: str, contract: ContractData) -> None:
        """"""
        underlying: Optional[UnderlyingData] = self.underlyings.get(contract.vt_symbol, None)
        if not underlying:
            underlying = UnderlyingData(contract)
            underlying.set_portfolio(self)
            self.underlyings[contract.vt_symbol] = underlying

        chain: ChainData = self.get_chain(chain_symbol)
        chain.set_underlying(underlying)

        # Add to active dict
        self.chains[chain_symbol] = chain

        for option in chain.options.values():
            self.options[option.vt_symbol] = option

    def get_chain(self, chain_symbol: str) -> ChainData:
        """"""
        chain: Optional[ChainData] = self._chains.get(chain_symbol, None)

        if not chain:
            chain = ChainData(chain_symbol, self.event_engine)
            chain.set_portfolio(self)
            self._chains[chain_symbol] = chain

        return chain

    def add_option(self, contract: ContractData) -> None:
        """"""
        option: OptionData = OptionData(contract)
        option.set_portfolio(self)
        self._options[contract.vt_symbol] = option

        exchange_name: str = contract.exchange.value
        chain_symbol: str = f"{contract.option_underlying}.{exchange_name}"

        chain: ChainData = self.get_chain(chain_symbol)
        chain.add_option(option)

    def calculate_atm_price(self) -> None:
        """"""
        for chain in self.chains.values():
            chain.calculate_atm_price()


@lru_cache(maxsize=100)
def get_underlying_prefix(portfolio_name: str) -> str:
    """
    基于期权产品名称获取对应标的代码

    已知规则：
    "510050_O.SSE": "510050"
    "159919_O.SZSE": "159919"

    "IO.CFFEX": "IF",
    "HO.CFFEX": "IH",
    "MO.CFFEX": "IM",

    "i_o.DCE": "i",
    "cu_o.SHFE": "cu",
    "sc_o.INE": "sc",
    "SR.CZCE": "SR",
    "lc_o.CFEX": "lc"
    """
    # 上交所
    if portfolio_name.endswith("SSE"):
        return portfolio_name.replace("_O.SSE", "")
    # 深交所
    elif portfolio_name.endswith("SZSE"):
        return portfolio_name.replace("_O.SZSE", "")
    # 港交所
    elif portfolio_name.endswith("SEHK"):
        return portfolio_name.replace("_O.SEHK", "")
    # 美股
    elif portfolio_name.endswith("SMART"):
        return portfolio_name.replace("_O.SMART", "")
    # 中金所（特殊规则）
    elif portfolio_name.endswith("CFFEX"):
        d: dict = {
            "IO.CFFEX": "IF",
            "HO.CFFEX": "IH",
            "MO.CFFEX": "IM",
        }
        return d.get(portfolio_name, "")
    # 上期所
    elif portfolio_name.endswith("SHFE"):
        return portfolio_name.replace("_o.SHFE", "")
    # 能交所
    elif portfolio_name.endswith("INE"):
        return portfolio_name.replace("_o.INE", "")
    # 大商所
    elif portfolio_name.endswith("DCE"):
        return portfolio_name.replace("_o.DCE", "")
    # 郑商所
    elif portfolio_name.endswith("CZCE"):
        return portfolio_name.replace(".CZCE", "")
    # 广期所
    elif portfolio_name.endswith("GFEX"):
        return portfolio_name.replace("_o.GFEX", "")
    # 其他
    else:
        return ""
