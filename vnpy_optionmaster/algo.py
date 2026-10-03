"""电子眼期权交易算法。"""
from typing import TYPE_CHECKING

from vnpy.trader.object import TickData, OrderData, TradeData
from vnpy.trader.constant import Direction, Offset
from vnpy.trader.utility import round_to

from .base import OptionData, UnderlyingData

if TYPE_CHECKING:
    from .engine import OptionAlgoEngine


class ElectronicEyeAlgo:
    """对单只期权做定价和双边狙击的电子眼算法。"""

    def __init__(
        self,
        algo_engine: "OptionAlgoEngine",
        option: OptionData
    ) -> None:
        """绑定算法引擎和期权，并初始化定价、交易参数和活动委托。"""
        self.algo_engine: OptionAlgoEngine = algo_engine
        self.option: OptionData = option
        self.underlying: UnderlyingData = option.underlying
        self.pricetick: float = option.pricetick
        self.vt_symbol: str = option.vt_symbol

        # Parameters
        self.pricing_active: bool = False
        self.trading_active: bool = False

        self.price_spread: float = 0.0
        self.volatility_spread: float = 0.0

        self.long_allowed: bool = False
        self.short_allowed: bool = False

        self.max_pos: int = 0
        self.target_pos: int = 0
        self.max_order_size: int = 0

        # Variables
        self.long_active_orderids: set[str] = set()
        self.short_active_orderids: set[str] = set()

        self.algo_spread: float = 0.0
        self.ref_price: float = 0.0
        self.algo_bid_price: float = 0.0
        self.algo_ask_price: float = 0.0
        self.pricing_impv: float = 0.0

    def start_pricing(self, params: dict) -> bool:
        """已在定价时返回假，否则保存价格价差和隐波价差并启动定价。"""
        if self.pricing_active:
            return False

        self.price_spread = params["price_spread"]
        self.volatility_spread = params["volatility_spread"]

        self.pricing_active = True
        self.put_status_event()
        self.calculate_price()
        self.write_log("启动定价")

        return True

    def stop_pricing(self) -> bool:
        """未在定价或仍在交易时返回假，否则清空定价结果并停止定价。"""
        if not self.pricing_active:
            return False

        if self.trading_active:
            return False

        self.pricing_active = False

        # Clear parameters
        self.algo_spread = 0.0
        self.ref_price = 0.0
        self.algo_bid_price = 0.0
        self.algo_ask_price = 0.0
        self.pricing_impv = 0.0

        self.put_status_event()
        self.put_pricing_event()
        self.write_log("停止定价")

        return True

    def start_trading(self, params: dict) -> bool:
        """已在交易、尚未定价或最大委托数量为 0 时返回假，否则保存交易参数并启动交易。"""
        if self.trading_active:
            return False

        if not self.pricing_active:
            self.write_log("请先启动定价")
            return False

        self.long_allowed = params["long_allowed"]
        self.short_allowed = params["short_allowed"]
        self.max_pos = params["max_pos"]
        self.target_pos = params["target_pos"]
        self.max_order_size = params["max_order_size"]

        if not self.max_order_size:
            self.write_log("请先设置最大委托数量")
            return False

        self.trading_active = True

        self.put_trading_event()
        self.put_status_event()
        self.write_log("启动交易")

        return True

    def stop_trading(self) -> bool:
        """未在交易时返回假，否则撤销多头和空头活动委托并停止交易。"""
        if not self.trading_active:
            return False

        self.trading_active = False

        self.cancel_long()
        self.cancel_short()

        self.put_status_event()
        self.put_trading_event()
        self.write_log("停止交易")

        return True

    def on_underlying_tick(self, tick: TickData) -> None:
        """标的行情到来时，定价开启则重算价格，交易开启则执行交易。"""
        if self.pricing_active:
            self.calculate_price()

        if self.trading_active:
            self.do_trading()

    def on_option_tick(self, tick: TickData) -> None:
        """期权行情到来时，交易开启则执行交易。"""
        if self.trading_active:
            self.do_trading()

    def on_order(self, order: OrderData) -> None:
        """委托不再活动时，从多头或空头活动委托中移除。"""
        if not order.is_active():
            if order.vt_orderid in self.long_active_orderids:
                self.long_active_orderids.remove(order.vt_orderid)
            elif order.vt_orderid in self.short_active_orderids:
                self.short_active_orderids.remove(order.vt_orderid)

    def on_trade(self, trade: TradeData) -> None:
        """把成交方向、开平、数量、价格和委托号写入日志。"""
        msg: str = (
            f"委托成交，{trade.direction} {trade.offset} {trade.volume}@{trade.price}，"
            f"委托号[{trade.vt_orderid}，成交号[{trade.vt_tradeid}]"
        )
        self.write_log(msg)

    def on_timer(self) -> None:
        """撤销尚未结束的多头和空头活动委托。"""
        if self.long_active_orderids:
            self.cancel_long()

        if self.short_active_orderids:
            self.cancel_short()

    def send_order(
        self,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: int
    ) -> str:
        """交给算法引擎发出委托，写入日志并返回委托号。"""
        vt_orderid: str = self.algo_engine.send_order(
            self,
            self.vt_symbol,
            direction,
            offset,
            price,
            volume
        )

        self.write_log(f"发出委托，{direction} {offset} {volume}@{price} [{vt_orderid}]")

        return vt_orderid

    def buy(self, price: float, volume: int) -> None:
        """买入开仓，并把返回的委托号记入多头活动委托。"""
        vt_orderid: str = self.send_order(Direction.LONG, Offset.OPEN, price, volume)
        self.long_active_orderids.add(vt_orderid)

    def sell(self, price: float, volume: int) -> None:
        """卖出平仓，并把返回的委托号记入空头活动委托。"""
        vt_orderid: str = self.send_order(Direction.SHORT, Offset.CLOSE, price, volume)
        self.short_active_orderids.add(vt_orderid)

    def short(self, price: float, volume: int) -> None:
        """卖出开仓，并把返回的委托号记入空头活动委托。"""
        vt_orderid: str = self.send_order(Direction.SHORT, Offset.OPEN, price, volume)
        self.short_active_orderids.add(vt_orderid)

    def cover(self, price: float, volume: int) -> None:
        """买入平仓，并把返回的委托号记入多头活动委托。"""
        vt_orderid: str = self.send_order(Direction.LONG, Offset.CLOSE, price, volume)
        self.long_active_orderids.add(vt_orderid)

    def send_long(self, price: float, volume: int) -> None:
        """没有空仓时买入开仓，空仓足够时买入平仓，否则先平掉空仓再买入剩余数量。"""
        option: OptionData = self.option

        if not option.short_pos:
            self.buy(price, volume)
        elif option.short_pos >= volume:
            self.cover(price, volume)
        else:
            self.cover(price, option.short_pos)
            self.buy(price, volume - option.short_pos)

    def send_short(self, price: float, volume: int) -> None:
        """没有多仓时卖出开仓，多仓足够时卖出平仓，否则先平掉多仓再卖出剩余数量。"""
        option: OptionData = self.option

        if not option.long_pos:
            self.short(price, volume)
        elif option.long_pos >= volume:
            self.sell(price, volume)
        else:
            self.sell(price, option.long_pos)
            self.short(price, volume - option.long_pos)

    def cancel_order(self, vt_orderid: str) -> None:
        """写入撤单日志，并交给算法引擎撤单。"""
        self.write_log(f"委托撤单：[{vt_orderid}]")
        self.algo_engine.cancel_order(vt_orderid)

    def cancel_long(self) -> None:
        """撤销全部多头活动委托。"""
        for vt_orderid in self.long_active_orderids:
            self.cancel_order(vt_orderid)

    def cancel_short(self) -> None:
        """撤销全部空头活动委托。"""
        for vt_orderid in self.short_active_orderids:
            self.cancel_order(vt_orderid)

    def check_long_finished(self) -> bool:
        """没有多头活动委托时返回真。"""
        if not self.long_active_orderids:
            return True

        return False

    def check_short_finished(self) -> bool:
        """没有空头活动委托时返回真。"""
        if not self.short_active_orderids:
            return True

        return False

    def calculate_price(self) -> None:
        """用定价隐含波动率计算参考价并按最小变动取整，价差取价格价差和隐波价差乘理论 vega 除以合约乘数的较大值。"""
        option: OptionData = self.option

        # Get ref price
        self.pricing_impv = option.pricing_impv
        ref_price: float = option.calculate_ref_price()
        self.ref_price = round_to(ref_price, self.pricetick)

        # Calculate spread
        algo_spread: float = max(
            self.price_spread,
            self.volatility_spread * option.theo_vega / option.size
        )
        half_spread: float = algo_spread / 2

        # Calculate bid/ask
        self.algo_bid_price = round_to(ref_price - half_spread, self.pricetick)
        self.algo_ask_price = round_to(ref_price + half_spread, self.pricetick)
        self.algo_spread = round_to(algo_spread, self.pricetick)

        self.put_pricing_event()

    def do_trading(self) -> None:
        """允许做多且没有多头活动委托时狙击买入，允许做空且没有空头活动委托时狙击卖出。"""
        if self.long_allowed and self.check_long_finished():
            self.snipe_long()

        if self.short_allowed and self.check_short_finished():
            self.snipe_short()

    def snipe_long(self) -> None:
        """无行情时返回；卖一价不高于算法买价且净持仓低于目标持仓加持仓范围时，按算法买价做多。"""
        option: OptionData = self.option
        tick: TickData | None = option.tick
        if not tick:
            return

        # Calculate volume left to trade
        pos_up_limit: int = self.target_pos + self.max_pos
        volume_left: int = pos_up_limit - option.net_pos

        # Check price
        if volume_left > 0 and tick.ask_price_1 <= self.algo_bid_price:
            volume = min(
                volume_left,
                tick.ask_volume_1,
                self.max_order_size
            )

            self.send_long(self.algo_bid_price, volume)     # type: ignore

    def snipe_short(self) -> None:
        """无行情时返回；买一价不低于算法卖价且净持仓高于目标持仓减持仓范围时，按算法卖价做空。"""
        option: OptionData = self.option
        tick: TickData | None = option.tick
        if not tick:
            return

        # Calculate volume left to trade
        pos_down_limit: int = self.target_pos - self.max_pos
        volume_left: int = option.net_pos - pos_down_limit

        # Check price
        if volume_left > 0 and tick.bid_price_1 >= self.algo_ask_price:
            volume = min(
                volume_left,
                tick.bid_volume_1,
                self.max_order_size
            )

            self.send_short(self.algo_ask_price, volume)     # type: ignore

    def put_pricing_event(self) -> None:
        """向算法引擎推送定价事件。"""
        self.algo_engine.put_algo_pricing_event(self)

    def put_trading_event(self) -> None:
        """向算法引擎推送交易事件。"""
        self.algo_engine.put_algo_trading_event(self)

    def put_status_event(self) -> None:
        """向算法引擎推送状态事件。"""
        self.algo_engine.put_algo_status_event(self)

    def write_log(self, msg: str) -> None:
        """把日志交给算法引擎写入。"""
        self.algo_engine.write_algo_log(self, msg)
