from typing import List, Dict, Set, Union, Optional
from copy import copy
from collections import defaultdict

from vnpy.event import Event, EventEngine
from vnpy.trader.ui import QtWidgets, QtCore, QtGui
from vnpy.trader.ui.widget import COLOR_BID, COLOR_ASK, COLOR_BLACK
from vnpy.trader.event import (
    EVENT_TICK, EVENT_TRADE, EVENT_POSITION, EVENT_TIMER
)
from vnpy.trader.object import TickData, TradeData, PositionData
from vnpy.trader.utility import round_to

from ..engine import OptionEngine
from ..base import UnderlyingData, OptionData, ChainData, PortfolioData, InstrumentData


COLOR_WHITE = QtGui.QColor("white")
COLOR_POS = QtGui.QColor("yellow")
COLOR_GREEKS = QtGui.QColor("cyan")


class MonitorCell(QtWidgets.QTableWidgetItem):
    """"""

    def __init__(self, text: str = "", vt_symbol: str = "") -> None:
        """"""
        super().__init__(text)

        self.vt_symbol: str = vt_symbol

        self.setTextAlignment(QtCore.Qt.AlignCenter)


class IndexCell(MonitorCell):
    """"""

    def __init__(self, text: str = "", vt_symbol: str = "") -> None:
        """"""
        super().__init__(text, vt_symbol)

        self.setForeground(COLOR_BLACK)
        self.setBackground(COLOR_WHITE)


class BidCell(MonitorCell):
    """"""

    def __init__(self, text: str = "", vt_symbol: str = "") -> None:
        """"""
        super().__init__(text, vt_symbol)

        self.setForeground(COLOR_BID)


class AskCell(MonitorCell):
    """"""

    def __init__(self, text: str = "", vt_symbol: str = "") -> None:
        """"""
        super().__init__(text, vt_symbol)

        self.setForeground(COLOR_ASK)


class PosCell(MonitorCell):
    """"""

    def __init__(self, text: str = "", vt_symbol: str = "") -> None:
        """"""
        super().__init__(text, vt_symbol)

        self.setForeground(COLOR_POS)


class GreeksCell(MonitorCell):
    """"""

    def __init__(self, text: str = "", vt_symbol: str = "") -> None:
        """"""
        super().__init__(text, vt_symbol)

        self.setForeground(COLOR_GREEKS)


class MonitorTable(QtWidgets.QTableWidget):
    """"""

    def __init__(self) -> None:
        """"""
        super().__init__()

        self.init_menu()

    def init_menu(self) -> None:
        """
        Create right click menu.
        """
        self.menu: QtWidgets.QMenu = QtWidgets.QMenu(self)

        resize_action = QtGui.QAction("调整列宽", self)
        resize_action.triggered.connect(self.resizeColumnsToContents)
        self.menu.addAction(resize_action)

    def contextMenuEvent(self, event) -> None:
        """
        Show menu with right click.
        """
        self.menu.popup(QtGui.QCursor.pos())


class OptionMarketMonitor(MonitorTable):
    """"""
    signal_tick: QtCore.Signal = QtCore.Signal(Event)
    signal_trade: QtCore.Signal = QtCore.Signal(Event)
    signal_position: QtCore.Signal = QtCore.Signal(Event)

    headers: List[Dict] = [
        {"name": "symbol", "display": "代码", "cell": MonitorCell},
        {"name": "theo_vega", "display": "Vega", "cell": GreeksCell},
        {"name": "theo_theta", "display": "Theta", "cell": GreeksCell},
        {"name": "theo_gamma", "display": "Gamma", "cell": GreeksCell},
        {"name": "theo_delta", "display": "Delta", "cell": GreeksCell},
        {"name": "open_interest", "display": "持仓量", "cell": MonitorCell},
        {"name": "volume", "display": "成交量", "cell": MonitorCell},
        {"name": "bid_impv", "display": "买隐波", "cell": BidCell},
        {"name": "bid_volume", "display": "买量", "cell": BidCell},
        {"name": "bid_price", "display": "买价", "cell": BidCell},
        {"name": "ask_price", "display": "卖价", "cell": AskCell},
        {"name": "ask_volume", "display": "卖量", "cell": AskCell},
        {"name": "ask_impv", "display": "卖隐波", "cell": AskCell},
        {"name": "net_pos", "display": "净持仓", "cell": PosCell},
    ]

    def __init__(self, option_engine: OptionEngine, portfolio_name: str) -> None:
        """"""
        super().__init__()

        self.option_engine: OptionEngine = option_engine
        self.event_engine: EventEngine = option_engine.event_engine
        self.portfolio_name: str = portfolio_name

        self.cells: Dict[str, Dict] = {}
        self.option_symbols: Set[str] = set()
        self.underlying_option_map: Dict[str, List] = defaultdict(list)

        self.init_ui()
        self.register_event()

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle("T型报价")
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(self.NoEditTriggers)

        # Store option and underlying symbols
        portfolio: PortfolioData = self.option_engine.get_portfolio(self.portfolio_name)

        for option in portfolio.options.values():
            self.option_symbols.add(option.vt_symbol)
            self.underlying_option_map[option.underlying.vt_symbol].append(option.vt_symbol)

        # Get greeks decimals precision
        self.greeks_precision: str = f"{portfolio.precision}f"

        # Set table row and column numbers
        row_count: int = 0
        for chain in portfolio.chains.values():
            row_count += (1 + len(chain.indexes))
        self.setRowCount(row_count)

        column_count: int = len(self.headers) * 2 + 1
        self.setColumnCount(column_count)

        call_labels: list = [d["display"] for d in self.headers]
        put_labels: list = copy(call_labels)
        put_labels.reverse()
        labels: list = call_labels + ["行权价"] + put_labels
        self.setHorizontalHeaderLabels(labels)

        # Init cells
        strike_column: int = len(self.headers)
        current_row: int = 0

        chain_symbols: list = list(portfolio.chains.keys())
        chain_symbols.sort()

        for chain_symbol in chain_symbols:
            chain: ChainData = portfolio.get_chain(chain_symbol)

            self.setItem(
                current_row,
                strike_column,
                IndexCell(chain.chain_symbol.split(".")[0])
            )

            for index in chain.indexes:
                call: OptionData = chain.calls[index]
                put: OptionData = chain.puts[index]

                current_row += 1

                # Call cells
                call_cells: dict = {}

                for column, d in enumerate(self.headers):
                    value = getattr(call, d["name"], "")
                    cell = d["cell"](
                        text=str(value),
                        vt_symbol=call.vt_symbol
                    )
                    self.setItem(current_row, column, cell)
                    call_cells[d["name"]] = cell

                self.cells[call.vt_symbol] = call_cells

                # Put cells
                put_cells: dict = {}
                put_headers: list = copy(self.headers)
                put_headers.reverse()

                for column, d in enumerate(put_headers):
                    column += (strike_column + 1)
                    value = getattr(put, d["name"], "")
                    cell = d["cell"](
                        text=str(value),
                        vt_symbol=put.vt_symbol
                    )
                    self.setItem(current_row, column, cell)
                    put_cells[d["name"]] = cell

                self.cells[put.vt_symbol] = put_cells

                # Strike cell
                index_cell: IndexCell = IndexCell(str(call.chain_index))
                self.setItem(current_row, strike_column, index_cell)

            # Move to next row
            current_row += 1

    def register_event(self) -> None:
        """"""
        self.signal_tick.connect(self.process_tick_event)
        self.signal_trade.connect(self.process_trade_event)
        self.signal_position.connect(self.process_position_event)

        self.event_engine.register(EVENT_TICK, self.signal_tick.emit)
        self.event_engine.register(EVENT_TRADE, self.signal_trade.emit)
        self.event_engine.register(EVENT_POSITION, self.signal_position.emit)

    def process_tick_event(self, event: Event) -> None:
        """"""
        tick: TickData = event.data

        if tick.vt_symbol in self.option_symbols:
            self.update_price(tick.vt_symbol)
            self.update_impv(tick.vt_symbol)
            self.update_greeks(tick.vt_symbol)
        elif tick.vt_symbol in self.underlying_option_map:
            option_symbols: list = self.underlying_option_map[tick.vt_symbol]

            for vt_symbol in option_symbols:
                self.update_impv(vt_symbol)
                self.update_greeks(vt_symbol)

    def process_trade_event(self, event: Event) -> None:
        """"""
        trade: TradeData = event.data
        self.update_pos(trade.vt_symbol)

    def process_position_event(self, event: Event) -> None:
        """"""
        position: PositionData = event.data
        self.update_pos(position.vt_symbol)

    def update_pos(self, vt_symbol: str) -> None:
        """"""
        option_cells: Optional[dict] = self.cells.get(vt_symbol, None)
        if not option_cells:
            return

        option: InstrumentData = self.option_engine.get_instrument(vt_symbol)

        option_cells["net_pos"].setText(str(option.net_pos))

    def update_price(self, vt_symbol: str) -> None:
        """"""
        option_cells: Optional[dict] = self.cells.get(vt_symbol, None)
        if not option_cells:
            return

        option: InstrumentData = self.option_engine.get_instrument(vt_symbol)
        tick: TickData = option.tick
        option_cells["bid_price"].setText(f'{tick.bid_price_1:0.4f}')
        option_cells["bid_volume"].setText(str(tick.bid_volume_1))
        option_cells["ask_price"].setText(f'{tick.ask_price_1:0.4f}')
        option_cells["ask_volume"].setText(str(tick.ask_volume_1))
        option_cells["volume"].setText(str(tick.volume))
        option_cells["open_interest"].setText(str(tick.open_interest))

    def update_impv(self, vt_symbol: str) -> None:
        """"""
        option_cells: Optional[dict] = self.cells.get(vt_symbol, None)
        if not option_cells:
            return

        option: InstrumentData = self.option_engine.get_instrument(vt_symbol)
        option_cells["bid_impv"].setText(f"{option.bid_impv * 100:.2f}")
        option_cells["ask_impv"].setText(f"{option.ask_impv * 100:.2f}")

    def update_greeks(self, vt_symbol: str) -> None:
        """"""
        option_cells: Optional[dict] = self.cells.get(vt_symbol, None)
        if not option_cells:
            return

        option: InstrumentData = self.option_engine.get_instrument(vt_symbol)

        option_cells["theo_delta"].setText(f"{option.theo_delta:.{self.greeks_precision}}")
        option_cells["theo_gamma"].setText(f"{option.theo_gamma:.{self.greeks_precision}}")
        option_cells["theo_theta"].setText(f"{option.theo_theta:.{self.greeks_precision}}")
        option_cells["theo_vega"].setText(f"{option.theo_vega:.{self.greeks_precision}}")


class OptionGreeksMonitor(MonitorTable):
    """"""
    signal_tick: QtCore.Signal = QtCore.Signal(Event)
    signal_trade: QtCore.Signal = QtCore.Signal(Event)
    signal_position: QtCore.Signal = QtCore.Signal(Event)

    headers: List[Dict] = [
        {"name": "long_pos", "display": "多仓", "cell": PosCell},
        {"name": "short_pos", "display": "空仓", "cell": PosCell},
        {"name": "net_pos", "display": "净仓", "cell": PosCell},
        {"name": "pos_delta", "display": "Delta", "cell": GreeksCell},
        {"name": "pos_gamma", "display": "Gamma", "cell": GreeksCell},
        {"name": "pos_theta", "display": "Theta", "cell": GreeksCell},
        {"name": "pos_vega", "display": "Vega", "cell": GreeksCell}
    ]

    ROW_DATA = Union[OptionData, UnderlyingData, ChainData, PortfolioData]

    def __init__(self, option_engine: OptionEngine, portfolio_name: str) -> None:
        """"""
        super().__init__()

        self.option_engine: OptionEngine = option_engine
        self.event_engine: EventEngine = option_engine.event_engine
        self.portfolio_name: str = portfolio_name

        self.cells: Dict[tuple, Dict] = {}
        self.option_symbols: Set[str] = set()
        self.underlying_option_map: Dict[str, List] = defaultdict(list)

        self.init_ui()
        self.register_event()

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle("希腊值风险")
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(self.NoEditTriggers)

        # Store option and underlying symbols
        portfolio: PortfolioData = self.option_engine.get_portfolio(self.portfolio_name)

        for option in portfolio.options.values():
            self.option_symbols.add(option.vt_symbol)
            self.underlying_option_map[option.underlying.vt_symbol].append(option.vt_symbol)

        # Get greeks decimals precision
        self.greeks_precision: str = f"{portfolio.precision}f"

        # Set table row and column numbers
        row_count: int = 2

        row_count += (len(portfolio.underlyings) + 1)

        row_count += (len(portfolio.chains) + 1)

        for chain in portfolio.chains.values():
            row_count += len(chain.options)

        self.setRowCount(row_count)

        column_count: int = len(self.headers) + 2
        self.setColumnCount(column_count)

        labels: list = ["类别", "代码"] + [d["display"] for d in self.headers]
        self.setHorizontalHeaderLabels(labels)

        # Init cells
        row_settings: list = []
        row_settings.append((self.portfolio_name, "组合"))
        row_settings.append(None)

        underlying_symbols: list = list(portfolio.underlyings.keys())
        underlying_symbols.sort()
        for underlying_symbol in underlying_symbols:
            row_settings.append((underlying_symbol, "标的"))
        row_settings.append(None)

        chain_symbols: list = list(portfolio.chains.keys())
        chain_symbols.sort()
        for chain_symbol in chain_symbols:
            row_settings.append((chain_symbol, "期权链"))
        row_settings.append(None)

        option_symbols: list = list(portfolio.options.keys())
        option_symbols.sort()
        for option_symbol in option_symbols:
            row_settings.append((option_symbol, "期权"))

        for row, row_key in enumerate(row_settings):
            if not row_key:
                continue
            row_name, type_name = row_key

            type_cell: MonitorCell = MonitorCell(type_name)
            self.setItem(row, 0, type_cell)

            name = row_name.split(".")[0]
            name_cell: MonitorCell = MonitorCell(name)
            self.setItem(row, 1, name_cell)

            row_cells: dict = {}
            for column, d in enumerate(self.headers):
                cell = d["cell"]()
                self.setItem(row, column + 2, cell)
                row_cells[d["name"]] = cell
            self.cells[row_key] = row_cells

            if row_name != self.portfolio_name:
                self.hideRow(row)

        self.resizeColumnToContents(0)

    def register_event(self) -> None:
        """"""
        self.signal_tick.connect(self.process_tick_event)
        self.signal_trade.connect(self.process_trade_event)
        self.signal_position.connect(self.process_position_event)

        self.event_engine.register(EVENT_TICK, self.signal_tick.emit)
        self.event_engine.register(EVENT_TRADE, self.signal_trade.emit)
        self.event_engine.register(EVENT_POSITION, self.signal_position.emit)

    def process_tick_event(self, event: Event) -> None:
        """"""
        tick: TickData = event.data

        if tick.vt_symbol not in self.underlying_option_map:
            return

        self.update_underlying_tick(tick.vt_symbol)

    def process_trade_event(self, event: Event) -> None:
        """"""
        trade: TradeData = event.data
        if trade.vt_symbol not in self.option_symbols:
            return

        self.update_pos(trade.vt_symbol)

    def process_position_event(self, event: Event) -> None:
        """"""
        position: PositionData = event.data
        if position.vt_symbol not in self.option_symbols:
            return

        self.update_pos(position.vt_symbol)

    def update_underlying_tick(self, vt_symbol: str) -> None:
        """"""
        underlying: InstrumentData = self.option_engine.get_instrument(vt_symbol)
        self.update_row(vt_symbol, "标的", underlying)

        for chain in underlying.chains.values():
            self.update_row(chain.chain_symbol, "期权链", chain)

            for option in chain.options.values():
                self.update_row(option.vt_symbol, "期权", option)

        portfolio: PositionData = underlying.portfolio
        self.update_row(portfolio.name, "组合", portfolio)

    def update_pos(self, vt_symbol: str) -> None:
        """"""
        instrument: InstrumentData = self.option_engine.get_instrument(vt_symbol)
        if isinstance(instrument, OptionData):
            self.update_row(vt_symbol, "期权", instrument)
        else:
            self.update_row(vt_symbol, "标的", instrument)

        # For option, greeks of chain also needs to be updated.
        if isinstance(instrument, OptionData):
            chain: ChainData = instrument.chain
            self.update_row(chain.chain_symbol, "期权链", chain)

        portfolio: PortfolioData = instrument.portfolio
        self.update_row(portfolio.name, "组合", portfolio)

    def update_row(self, row_name: str, type_name: str, row_data: ROW_DATA) -> None:
        """"""
        row_key: tuple = (row_name, type_name)
        row_cells: dict = self.cells[row_key]
        row: int = self.row(row_cells["long_pos"])

        # Hide rows with no existing position
        if not row_data.long_pos and not row_data.short_pos:
            if row_name != self.portfolio_name:
                self.hideRow(row)
            return

        self.showRow(row)

        row_cells["long_pos"].setText(f"{row_data.long_pos}")
        row_cells["short_pos"].setText(f"{row_data.short_pos}")
        row_cells["net_pos"].setText(f"{row_data.net_pos}")
        row_cells["pos_delta"].setText(f"{row_data.pos_delta:.{self.greeks_precision}}")

        if not isinstance(row_data, UnderlyingData):
            row_cells["pos_gamma"].setText(f"{row_data.pos_gamma:.{self.greeks_precision}}")
            row_cells["pos_theta"].setText(f"{row_data.pos_theta:.{self.greeks_precision}}")
            row_cells["pos_vega"].setText(f"{row_data.pos_vega:.{self.greeks_precision}}")


class OptionChainMonitor(MonitorTable):
    """"""
    signal_timer: QtCore.Signal = QtCore.Signal(Event)

    def __init__(self, option_engine: OptionEngine, portfolio_name: str):
        """"""
        super().__init__()

        self.option_engine: OptionEngine = option_engine
        self.event_engine: EventEngine = option_engine.event_engine
        self.portfolio_name: str = portfolio_name

        self.cells: Dict[str, Dict] = {}

        self.init_ui()
        self.register_event()

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle("期权链跟踪")
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(self.NoEditTriggers)

        # Store option and underlying symbols
        portfolio: PortfolioData = self.option_engine.get_portfolio(self.portfolio_name)

        # Set table row and column numbers
        self.setRowCount(len(portfolio.chains))

        labels: list = ["期权链", "剩余交易日", "标的物", "升贴水"]
        self.setColumnCount(len(labels))
        self.setHorizontalHeaderLabels(labels)

        # Init cells
        chain_symbols: list = list(portfolio.chains.keys())
        chain_symbols.sort()

        for row, chain_symbol in enumerate(chain_symbols):
            chain: ChainData = portfolio.chains[chain_symbol]
            adjustment_cell: MonitorCell = MonitorCell()
            underlying_cell: MonitorCell = MonitorCell()

            self.setItem(row, 0, MonitorCell(chain.chain_symbol.split(".")[0]))
            self.setItem(row, 1, MonitorCell(str(chain.days_to_expiry)))
            self.setItem(row, 2, underlying_cell)
            self.setItem(row, 3, adjustment_cell)

            self.cells[chain.chain_symbol] = {
                "underlying": underlying_cell,
                "adjustment": adjustment_cell
            }

        # Additional table adjustment
        horizontal_header: QtWidgets.QHeaderView = self.horizontalHeader()
        horizontal_header.setSectionResizeMode(horizontal_header.Stretch)

    def register_event(self) -> None:
        """"""
        self.signal_timer.connect(self.process_timer_event)

        self.event_engine.register(EVENT_TIMER, self.signal_timer.emit)

    def process_timer_event(self, event: Event) -> None:
        """"""
        portfolio: PortfolioData = self.option_engine.get_portfolio(self.portfolio_name)

        for chain in portfolio.chains.values():
            underlying: UnderlyingData = chain.underlying

            underlying_symbol: str = underlying.vt_symbol.split(".")[0]

            if chain.underlying_adjustment == float("inf"):
                continue

            if underlying.pricetick:
                adjustment = round_to(chain.underlying_adjustment, underlying.pricetick)
            else:
                adjustment = 0

            chain_cells: dict = self.cells[chain.chain_symbol]
            chain_cells["underlying"].setText(underlying_symbol)
            chain_cells["adjustment"].setText(str(adjustment))
