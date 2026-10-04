"""期权组合、期权链、标的和期权合约数据。"""
from datetime import datetime
from collections.abc import Callable
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


APP_NAME: str = "OptionMaster"

EVENT_OPTION_NEW_PORTFOLIO: str = "eOptionNewPortfolio"
EVENT_OPTION_ALGO_PRICING: str = "eOptionAlgoPricing"
EVENT_OPTION_ALGO_TRADING: str = "eOptionAlgoTrading"
EVENT_OPTION_ALGO_STATUS: str = "eOptionAlgoStatus"
EVENT_OPTION_ALGO_LOG: str = "eOptionAlgoLog"
EVENT_OPTION_RISK_NOTICE: str = "eOptionRiskNotice"


class InstrumentData:
    """保存合约代码、行情和多空持仓的数据。"""

    def __init__(self, contract: ContractData) -> None:
        """从合约复制代码、最小变动、最小数量和合约乘数，并初始化持仓与中间价。"""
        self.symbol: str = contract.symbol
        self.exchange: Exchange = contract.exchange
        self.vt_symbol: str = contract.vt_symbol

        self.pricetick: float = contract.pricetick
        self.min_volume: float = contract.min_volume
        self.size: float = contract.size

        self.long_pos: int = 0
        self.short_pos: int = 0
        self.net_pos: int = 0
        self.mid_price: float = 0

        self.tick: TickData | None = None
        self.portfolio: PortfolioData

    def calculate_net_pos(self) -> None:
        """用多仓减去空仓更新净持仓。"""
        self.net_pos = self.long_pos - self.short_pos

    def update_tick(self, tick: TickData) -> None:
        """保存行情，并用买一价和卖一价的平均值作为中间价。"""
        self.tick = tick
        self.mid_price = (tick.bid_price_1 + tick.ask_price_1) / 2

    def update_trade(self, trade: TradeData) -> None:
        """多头开仓增加多仓，多头平仓减少空仓，空头开仓增加空仓，空头平仓减少多仓，然后重算净持仓。"""
        if trade.direction == Direction.LONG:
            if trade.offset == Offset.OPEN:
                self.long_pos += trade.volume       # type: ignore
            else:
                self.short_pos -= trade.volume      # type: ignore
        else:
            if trade.offset == Offset.OPEN:
                self.short_pos += trade.volume      # type: ignore
            else:
                self.long_pos -= trade.volume       # type: ignore
        self.calculate_net_pos()

    def update_holding(self, holding: PositionHolding) -> None:
        """用持仓换算器的多仓和空仓覆盖本地持仓，并重算净持仓。"""
        self.long_pos = holding.long_pos            # type: ignore
        self.short_pos = holding.short_pos          # type: ignore
        self.calculate_net_pos()

    def set_portfolio(self, portfolio: "PortfolioData") -> None:
        """设置所属组合。"""
        self.portfolio = portfolio


class OptionData(InstrumentData):
    """在合约数据上增加隐含波动率、定价和希腊值的期权。"""

    def __init__(self, contract: ContractData) -> None:
        """初始化行权价、认购认沽方向、到期时间和定价字段。"""
        super().__init__(contract)

        # Option contract features
        self.strike_price: float = contract.option_strike       # type: ignore
        self.chain_index: str = contract.option_index           # type: ignore

        self.option_type: int = 0
        if contract.option_type == OptionType.CALL:
            self.option_type = 1
        else:
            self.option_type = -1

        self.option_expiry: datetime = contract.option_expiry                       # type: ignore
        self.days_to_expiry: int = calculate_days_to_expiry(contract.option_expiry) # type: ignore
        self.time_to_expiry: float = self.days_to_expiry / ANNUAL_DAYS

        self.interest_rate: float = 0

        # Option portfolio related
        self.underlying: UnderlyingData
        self.chain: ChainData
        self.underlying_adjustment: float = 0

        # Pricing model
        self.calculate_price: Callable
        self.calculate_greeks: Callable
        self.calculate_impv: Callable

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
        """缺少行情、标的或标的中间价时返回，否则用标的中间价加调整量计算买价、卖价和中间价的隐含波动率。"""
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
            mid_price = ask_price
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
        """缺少标的、标的中间价或中间隐含波动率时返回，否则计算理论希腊值，delta 与 gamma 乘合约乘数，theta 再除以 240，vega 再除以 100。"""
        if not self.underlying:
            return

        underlying_price: float = self.underlying.mid_price
        if not underlying_price or not self.mid_impv:
            return
        underlying_price += self.underlying_adjustment

        delta: float
        gamma: float
        theta: float
        vega: float
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
        """有行情时用最新价乘合约乘数和净持仓得到持仓市值，并把理论希腊值乘净持仓。"""
        if self.tick:
            self.pos_value = self.tick.last_price * self.size * self.net_pos

        self.pos_delta = self.theo_delta * self.net_pos
        self.pos_gamma = self.theo_gamma * self.net_pos
        self.pos_theta = self.theo_theta * self.net_pos
        self.pos_vega = self.theo_vega * self.net_pos

    def calculate_ref_price(self) -> float:
        """用标的中间价加调整量和定价隐含波动率计算参考价。"""
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
        """更新行情后重算隐含波动率。"""
        super().update_tick(tick)

        self.calculate_option_impv()

    def update_trade(self, trade: TradeData) -> None:
        """更新成交后重算持仓希腊值。"""
        super().update_trade(trade)
        self.calculate_pos_greeks()

    def update_underlying_tick(self, underlying_adjustment: float) -> None:
        """记下标的调整量，并重算隐含波动率、理论希腊值和持仓希腊值。"""
        self.underlying_adjustment = underlying_adjustment

        self.calculate_option_impv()
        self.calculate_theo_greeks()
        self.calculate_pos_greeks()

    def set_chain(self, chain: "ChainData") -> None:
        """设置所属期权链。"""
        self.chain = chain

    def set_underlying(self, underlying: "UnderlyingData") -> None:
        """设置标的合约。"""
        self.underlying = underlying

    def set_interest_rate(self, interest_rate: float) -> None:
        """设置利率。"""
        self.interest_rate = interest_rate

    def set_pricing_model(self, pricing_model: ModuleType) -> None:
        """绑定定价模型的希腊值、隐含波动率和价格函数。"""
        self.calculate_greeks = pricing_model.calculate_greeks
        self.calculate_impv = pricing_model.calculate_impv
        self.calculate_price = pricing_model.calculate_price


class UnderlyingData(InstrumentData):
    """带理论 delta 和期权链的标的数据。"""

    def __init__(self, contract: ContractData) -> None:
        """初始化标的，理论 delta 取合约乘数。"""
        super().__init__(contract)

        self.theo_delta: float = self.size                  # 标的物理论Delta固定为1
        self.pos_delta: float = 0
        self.chains: dict[str, ChainData] = {}

    def add_chain(self, chain: "ChainData") -> None:
        """按链代码登记期权链。"""
        self.chains[chain.chain_symbol] = chain

    def update_tick(self, tick: TickData) -> None:
        """更新标的行情，通知各期权链，并重算持仓 delta。"""
        super().update_tick(tick)

        chain: ChainData
        for chain in self.chains.values():
            chain.update_underlying_tick()

        self.calculate_pos_greeks()

    def update_trade(self, trade: TradeData) -> None:
        """更新标的成交后重算持仓 delta。"""
        super().update_trade(trade)

        self.calculate_pos_greeks()

    def calculate_pos_greeks(self) -> None:
        """用理论 delta 乘净持仓更新持仓 delta。"""
        self.pos_delta = self.theo_delta * self.net_pos


class ChainData:
    """同一标的月份的认购和认沽组成的期权链。"""

    def __init__(self, chain_symbol: str, event_engine: EventEngine) -> None:
        """初始化期权链的持仓、希腊值、合约容器和平值字段。"""
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

        self.underlying: UnderlyingData

        self.options: dict[str, OptionData] = {}
        self.calls: dict[str, OptionData] = {}
        self.puts: dict[str, OptionData] = {}

        self.portfolio: PortfolioData

        self.indexes: list[str] = []
        self.atm_price: float = 0
        self.atm_index: str = ""
        self.underlying_adjustment: float = 0
        self.days_to_expiry: int = 0

        self.use_synthetic: bool = False

    def add_option(self, option: OptionData) -> None:
        """登记期权并按行权索引排序，认购和认沽分开存放，剩余交易日改为该期权的剩余交易日。"""
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
        """先清零，再汇总净持仓非零的期权仓位和希腊值。"""
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
        option: OptionData
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
        """更新该期权行情；合成标的尚无平值时先计算平值，平值期权再刷新合成价。"""
        option: OptionData = self.options[tick.vt_symbol]
        option.update_tick(tick)

        if self.use_synthetic:
            if not self.atm_index:
                self.calculate_atm_price()

            if option.chain_index == self.atm_index:
                self.update_synthetic_price()

    def update_underlying_tick(self) -> None:
        """非合成标的时重算调整量，然后用该调整量更新链上全部期权并重算持仓希腊值。"""
        if not self.use_synthetic:
            self.calculate_underlying_adjustment()

        option: OptionData
        for option in self.options.values():
            option.update_underlying_tick(self.underlying_adjustment)

        self.calculate_pos_greeks()

    def update_trade(self, trade: TradeData) -> None:
        """先扣掉该期权的旧仓位和希腊值，更新成交后再加回新值。"""
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
        """绑定标的并写到链上每个期权；标的交易所为本地时改为使用合成期货。"""
        underlying.add_chain(self)
        self.underlying = underlying

        option: OptionData
        for option in self.options.values():
            option.set_underlying(underlying)

        if underlying.exchange == Exchange.LOCAL:
            self.use_synthetic = True

    def set_interest_rate(self, interest_rate: float) -> None:
        """把利率写到链上每个期权。"""
        option: OptionData
        for option in self.options.values():
            option.set_interest_rate(interest_rate)

    def set_pricing_model(self, pricing_model: ModuleType) -> None:
        """把定价模型写到链上每个期权。"""
        option: OptionData
        for option in self.options.values():
            option.set_pricing_model(pricing_model)

    def set_portfolio(self, portfolio: "PortfolioData") -> None:
        """把组合写到链上每个期权。"""
        option: OptionData
        for option in self.options.values():
            option.set_portfolio(portfolio)

    def calculate_atm_price(self) -> None:
        """在买卖价齐全的行权价中，取认购与认沽中间价相差最小者作为平值。"""
        min_diff: float = 0
        atm_price: float = 0
        atm_index: str = ""

        index: str
        call: OptionData
        for index, call in self.calls.items():
            put: OptionData = self.puts[index]

            call_tick: TickData | None = call.tick
            if not call_tick or not call_tick.bid_price_1 or not call_tick.ask_price_1:
                continue

            put_tick: TickData | None = put.tick
            if not put_tick or not put_tick.bid_price_1 or not put_tick.ask_price_1:
                continue

            call_mid_price: float = (call_tick.ask_price_1 + call_tick.bid_price_1) / 2
            put_mid_price: float = (put_tick.ask_price_1 + put_tick.bid_price_1) / 2

            diff: float = abs(call_mid_price - put_mid_price)

            if not min_diff or diff < min_diff:
                min_diff = diff
                atm_price = call.strike_price
                atm_index = call.chain_index

        self.atm_price = atm_price
        self.atm_index = atm_index

    def calculate_underlying_adjustment(self) -> None:
        """没有平值价格时返回，否则用平值认购中间价减认沽中间价加行权价，再减去标的中间价。"""
        if not self.atm_price:
            return

        atm_call: OptionData = self.calls[self.atm_index]
        atm_put: OptionData = self.puts[self.atm_index]

        call_price: float = atm_call.mid_price
        put_price: float = atm_put.mid_price

        synthetic_price: float = call_price - put_price + self.atm_price
        self.underlying_adjustment = synthetic_price - self.underlying.mid_price

    def update_synthetic_price(self) -> None:
        """用平值认购中间价减认沽中间价加行权价更新标的中间价，更新链上期权并推送合成行情。"""
        call: OptionData = self.calls[self.atm_index]
        put: OptionData = self.puts[self.atm_index]

        self.underlying.mid_price = call.mid_price - put.mid_price + self.atm_price
        self.update_underlying_tick()

        # 推送合成期货的行情
        symbol: str
        exchange: Exchange
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
    """管理期权、期权链和标的的组合数据。"""

    def __init__(self, name: str, event_engine: EventEngine) -> None:
        """初始化组合名称、持仓、希腊值精度，以及全部合约和活跃合约容器。"""
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
        self._options: dict[str, OptionData] = {}
        self._chains: dict[str, ChainData] = {}

        # Active instrument
        self.options: dict[str, OptionData] = {}
        self.chains: dict[str, ChainData] = {}
        self.underlyings: dict[str, UnderlyingData] = {}

        # Greeks decimals precision
        self.precision: int = 0

    def calculate_pos_greeks(self) -> None:
        """汇总标的的持仓 delta，以及各活跃期权链的仓位和希腊值。"""
        self.long_pos = 0
        self.short_pos = 0
        self.net_pos = 0

        self.pos_value: float = 0.0
        self.pos_delta = 0
        self.pos_gamma = 0
        self.pos_theta = 0
        self.pos_vega = 0

        underlying: UnderlyingData
        for underlying in self.underlyings.values():
            self.pos_delta += underlying.pos_delta

        chain: ChainData
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
        """活跃期权的行情交给其所在链，活跃标的的行情交给标的，然后重算组合希腊值。"""
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
        """活跃期权的成交交给其所在链，活跃标的的成交交给标的，然后重算组合希腊值。"""
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
        """把利率写到每条活跃期权链。"""
        chain: ChainData
        for chain in self.chains.values():
            chain.set_interest_rate(interest_rate)

    def set_pricing_model(self, pricing_model: ModuleType) -> None:
        """把定价模型写到每条活跃期权链。"""
        chain: ChainData
        for chain in self.chains.values():
            chain.set_pricing_model(pricing_model)

    def set_precision(self, precision: int) -> None:
        """设置希腊值小数位数。"""
        self.precision = precision

    def set_chain_underlying(self, chain_symbol: str, contract: ContractData) -> None:
        """创建或复用标的并绑定到期权链，再把该链及其期权标为活跃。"""
        underlying: UnderlyingData | None = self.underlyings.get(contract.vt_symbol, None)
        if not underlying:
            underlying = UnderlyingData(contract)
            underlying.set_portfolio(self)
            self.underlyings[contract.vt_symbol] = underlying

        chain: ChainData = self.get_chain(chain_symbol)
        chain.set_underlying(underlying)

        # Add to active dict
        self.chains[chain_symbol] = chain

        option: OptionData
        for option in chain.options.values():
            self.options[option.vt_symbol] = option

    def get_chain(self, chain_symbol: str) -> ChainData:
        """按代码获取期权链，没有则创建并记入全部链。"""
        chain: ChainData | None = self._chains.get(chain_symbol, None)

        if not chain:
            chain = ChainData(chain_symbol, self.event_engine)
            chain.set_portfolio(self)
            self._chains[chain_symbol] = chain

        return chain

    def add_option(self, contract: ContractData) -> None:
        """按期权标的和交易所创建期权，并加入对应期权链。"""
        option: OptionData = OptionData(contract)
        option.set_portfolio(self)
        self._options[contract.vt_symbol] = option

        exchange_name: str = contract.exchange.value
        chain_symbol: str = f"{contract.option_underlying}.{exchange_name}"

        chain: ChainData = self.get_chain(chain_symbol)
        chain.add_option(option)

    def calculate_atm_price(self) -> None:
        """让每条活跃期权链重算平值价格。"""
        chain: ChainData
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
        prefix: str = d.get(portfolio_name, "")
        return prefix
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
    # 其他
    else:
        return ""
