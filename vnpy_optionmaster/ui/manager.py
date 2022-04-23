from typing import Dict, List, Tuple, Optional
from copy import copy
from functools import partial

from scipy import interpolate

from vnpy.event import Event, EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.ui import QtWidgets, QtCore, QtGui
from vnpy.trader.event import EVENT_TICK, EVENT_TIMER, EVENT_TRADE
from vnpy.trader.object import TickData, TradeData, LogData
from vnpy.trader.utility import save_json, load_json

from ..engine import OptionEngine, OptionAlgoEngine
from ..base import (
    EVENT_OPTION_ALGO_PRICING,
    EVENT_OPTION_ALGO_STATUS,
    EVENT_OPTION_ALGO_LOG,
    PortfolioData,
    ChainData,
    OptionData,
    InstrumentData
)
from .monitor import (
    MonitorCell, IndexCell, BidCell, AskCell, PosCell,
    COLOR_WHITE, COLOR_BLACK
)
from ..algo import ElectronicEyeAlgo


class AlgoSpinBox(QtWidgets.QSpinBox):
    """"""

    def __init__(self) -> None:
        """"""
        super().__init__()

        self.setMaximum(999999)
        self.setMinimum(-999999)
        self.setAlignment(QtCore.Qt.AlignCenter)

    def get_value(self) -> int:
        """"""
        return self.value()

    def set_value(self, value: int) -> None:
        """"""
        self.setValue(value)

    def update_status(self, active: bool) -> None:
        """"""
        self.setEnabled(not active)


class AlgoPositiveSpinBox(AlgoSpinBox):
    """"""

    def __init__(self) -> None:
        """"""
        super().__init__()

        self.setMinimum(0)


class AlgoDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """"""

    def __init__(self) -> None:
        """"""
        super().__init__()

        self.setDecimals(1)
        self.setMaximum(9999.9)
        self.setMinimum(0)
        self.setAlignment(QtCore.Qt.AlignCenter)

    def get_value(self) -> float:
        """"""
        return self.value()

    def set_value(self, value: float) -> None:
        """"""
        self.setValue(value)

    def update_status(self, active: bool) -> None:
        """"""
        self.setEnabled(not active)


class AlgoDirectionCombo(QtWidgets.QComboBox):
    """"""

    def __init__(self) -> None:
        """"""
        super().__init__()

        self.addItems([
            "双向",
            "做多",
            "做空"
        ])

    def get_value(self) -> Dict[str, bool]:
        """"""
        if self.currentText() == "双向":
            value: dict = {
                "long_allowed": True,
                "short_allowed": True
            }
        elif self.currentText() == "做多":
            value: dict = {
                "long_allowed": True,
                "short_allowed": False
            }
        else:
            value: dict = {
                "long_allowed": False,
                "short_allowed": True
            }

        return value

    def set_value(self, value: dict) -> None:
        """"""
        if value["long_allowed"] and value["short_allowed"]:
            self.setCurrentIndex(0)
        elif value["long_allowed"]:
            self.setCurrentIndex(1)
        else:
            self.setCurrentIndex(2)

    def update_status(self, active: bool) -> None:
        """"""
        self.setEnabled(not active)


class AlgoPricingButton(QtWidgets.QPushButton):
    """"""

    def __init__(self, vt_symbol: str, manager: "ElectronicEyeManager") -> None:
        """"""
        super().__init__()

        self.vt_symbol: str = vt_symbol
        self.manager: ElectronicEyeManager = manager

        self.active: bool = False
        self.setText("N")
        self.clicked.connect(self.on_clicked)

    def on_clicked(self) -> None:
        """"""
        if self.active:
            self.manager.stop_algo_pricing(self.vt_symbol)
        else:
            self.manager.start_algo_pricing(self.vt_symbol)

    def update_status(self, active: bool) -> None:
        """"""
        self.active = active

        if active:
            self.setText("Y")
        else:
            self.setText("N")


class AlgoTradingButton(QtWidgets.QPushButton):
    """"""

    def __init__(self, vt_symbol: str, manager: "ElectronicEyeManager") -> None:
        """"""
        super().__init__()

        self.vt_symbol: str = vt_symbol
        self.manager: ElectronicEyeManager = manager

        self.active: bool = False
        self.setText("N")
        self.clicked.connect(self.on_clicked)

    def on_clicked(self) -> None:
        """"""
        if self.active:
            self.manager.stop_algo_trading(self.vt_symbol)
        else:
            self.manager.start_algo_trading(self.vt_symbol)

    def update_status(self, active: bool) -> None:
        """"""
        self.active = active

        if active:
            self.setText("Y")
        else:
            self.setText("N")


class ElectronicEyeMonitor(QtWidgets.QTableWidget):
    """"""

    signal_tick: QtCore.Signal = QtCore.Signal(Event)
    signal_pricing: QtCore.Signal = QtCore.Signal(Event)
    signal_status: QtCore.Signal = QtCore.Signal(Event)
    signal_trade: QtCore.Signal = QtCore.Signal(Event)

    headers: List[Dict] = [
        {"name": "bid_volume", "display": "买量", "cell": BidCell},
        {"name": "bid_price", "display": "买价", "cell": BidCell},
        {"name": "ask_price", "display": "卖价", "cell": AskCell},
        {"name": "ask_volume", "display": "卖量", "cell": AskCell},
        {"name": "algo_bid_price", "display": "目标\n买价", "cell": BidCell},
        {"name": "algo_ask_price", "display": "目标\n卖价", "cell": AskCell},
        {"name": "algo_spread", "display": "价差", "cell": MonitorCell},
        {"name": "ref_price", "display": "理论价", "cell": MonitorCell},
        {"name": "pricing_impv", "display": "定价\n隐波", "cell": MonitorCell},
        {"name": "net_pos", "display": "净持仓", "cell": PosCell},

        {"name": "price_spread", "display": "价格\n价差", "cell": AlgoDoubleSpinBox},
        {"name": "volatility_spread", "display": "隐波\n价差", "cell": AlgoDoubleSpinBox},
        {"name": "max_pos", "display": "持仓\n范围", "cell": AlgoPositiveSpinBox},
        {"name": "target_pos", "display": "目标\n持仓", "cell": AlgoSpinBox},
        {"name": "max_order_size", "display": "最大\n委托", "cell": AlgoPositiveSpinBox},
        {"name": "direction", "display": "方向", "cell": AlgoDirectionCombo},
        {"name": "pricing_active", "display": "定价", "cell": AlgoPricingButton},
        {"name": "trading_active", "display": "交易", "cell": AlgoTradingButton},
    ]

    def __init__(self, option_engine: OptionEngine, portfolio_name: str) -> None:
        """"""
        super().__init__()

        self.option_engine: OptionEngine = option_engine
        self.event_engine: EventEngine = option_engine.event_engine
        self.main_engine: MainEngine = option_engine.main_engine
        self.algo_engine: OptionAlgoEngine = option_engine.algo_engine
        self.portfolio_name: str = portfolio_name
        self.setting_filename: str = f"{portfolio_name}_electronic_eye.json"

        self.cells: Dict[str, Dict] = {}

        self.init_ui()
        self.register_event()
        self.load_setting()

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle("电子眼")
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(self.NoEditTriggers)

        # Set table row and column numbers
        portfolio: PortfolioData = self.option_engine.get_portfolio(self.portfolio_name)

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
                    cell_type = d["cell"]

                    if issubclass(cell_type, QtWidgets.QPushButton):
                        cell = cell_type(call.vt_symbol, self)
                    else:
                        cell = cell_type()

                    call_cells[d["name"]] = cell

                    if isinstance(cell, QtWidgets.QTableWidgetItem):
                        self.setItem(current_row, column, cell)
                    else:
                        self.setCellWidget(current_row, column, cell)

                self.cells[call.vt_symbol] = call_cells

                # Put cells
                put_cells: dict = {}
                put_headers: list = copy(self.headers)
                put_headers.reverse()

                for column, d in enumerate(put_headers):
                    column += (strike_column + 1)

                    cell_type = d["cell"]

                    if issubclass(cell_type, QtWidgets.QPushButton):
                        cell = cell_type(put.vt_symbol, self)
                    else:
                        cell = cell_type()

                    put_cells[d["name"]] = cell

                    if isinstance(cell, QtWidgets.QTableWidgetItem):
                        self.setItem(current_row, column, cell)
                    else:
                        self.setCellWidget(current_row, column, cell)

                self.cells[put.vt_symbol] = put_cells

                # Strike cell
                index_cell: IndexCell = IndexCell(str(call.chain_index))
                self.setItem(current_row, strike_column, index_cell)

            # Move to next row
            current_row += 1

        self.resizeColumnsToContents()

        # Update all net pos and tick cells
        for vt_symbol in self.cells.keys():
            self.update_net_pos(vt_symbol)

            tick: Optional[TickData] = self.main_engine.get_tick(vt_symbol)
            if tick:
                self.update_tick(tick)

    def load_setting(self) -> None:
        """"""
        fields: list = [
            "price_spread",
            "volatility_spread",
            "max_pos",
            "target_pos",
            "max_order_size",
            "direction"
        ]

        setting: dict = load_json(self.setting_filename)

        for vt_symbol, cells in self.cells.items():
            buf: Optional[dict] = setting.get(vt_symbol, None)
            if buf:
                for field in fields:
                    cells[field].set_value(buf[field])

    def save_setting(self) -> None:
        """"""
        fields: list = [
            "price_spread",
            "volatility_spread",
            "max_pos",
            "target_pos",
            "max_order_size",
            "direction"
        ]

        setting: dict = {}
        for vt_symbol, cells in self.cells.items():
            buf: dict = {}
            for field in fields:
                buf[field] = cells[field].get_value()
            setting[vt_symbol] = buf

        save_json(self.setting_filename, setting)

    def register_event(self) -> None:
        """"""
        self.signal_pricing.connect(self.process_pricing_event)
        self.signal_status.connect(self.process_status_event)
        self.signal_tick.connect(self.process_tick_event)
        self.signal_trade.connect(self.process_trade_event)

        self.event_engine.register(
            EVENT_OPTION_ALGO_PRICING,
            self.signal_pricing.emit
        )
        self.event_engine.register(
            EVENT_OPTION_ALGO_STATUS,
            self.signal_status.emit
        )
        self.event_engine.register(
            EVENT_TICK,
            self.signal_tick.emit
        )
        self.event_engine.register(
            EVENT_TRADE,
            self.signal_trade.emit
        )

    def process_tick_event(self, event: Event) -> None:
        """"""
        tick: TickData = event.data
        self.update_tick(tick)

    def update_tick(self, tick: TickData) -> None:
        """"""
        cells: Optional[dict] = self.cells.get(tick.vt_symbol, None)
        if not cells:
            return

        cells["bid_price"].setText(str(tick.bid_price_1))
        cells["ask_price"].setText(str(tick.ask_price_1))
        cells["bid_volume"].setText(str(tick.bid_volume_1))
        cells["ask_volume"].setText(str(tick.ask_volume_1))

    def process_status_event(self, event: Event) -> None:
        """"""
        algo: ElectronicEyeAlgo = event.data
        cells: dict = self.cells[algo.vt_symbol]

        cells["price_spread"].update_status(algo.pricing_active)
        cells["volatility_spread"].update_status(algo.pricing_active)
        cells["pricing_active"].update_status(algo.pricing_active)

        cells["max_pos"].update_status(algo.trading_active)
        cells["target_pos"].update_status(algo.trading_active)
        cells["max_order_size"].update_status(algo.trading_active)
        cells["direction"].update_status(algo.trading_active)
        cells["trading_active"].update_status(algo.trading_active)

    def process_pricing_event(self, event: Event) -> None:
        """"""
        algo: ElectronicEyeAlgo = event.data
        cells: dict = self.cells[algo.vt_symbol]

        if algo.ref_price:
            cells["algo_bid_price"].setText(str(algo.algo_bid_price))
            cells["algo_ask_price"].setText(str(algo.algo_ask_price))
            cells["algo_spread"].setText(str(algo.algo_spread))
            cells["ref_price"].setText(str(algo.ref_price))
            cells["pricing_impv"].setText(f"{algo.pricing_impv * 100:.2f}")
        else:
            cells["algo_bid_price"].setText("")
            cells["algo_ask_price"].setText("")
            cells["algo_spread"].setText("")
            cells["ref_price"].setText("")
            cells["pricing_impv"].setText("")

    def process_trade_event(self, event: Event) -> None:
        """"""
        trade: TradeData = event.data
        self.update_net_pos(trade.vt_symbol)

    def update_net_pos(self, vt_symbol: str) -> None:
        """"""
        cells: Optional[dict] = self.cells.get(vt_symbol, None)
        if not cells:
            return

        option: InstrumentData = self.option_engine.get_instrument(vt_symbol)
        cells["net_pos"].setText(str(option.net_pos))

    def start_algo_pricing(self, vt_symbol: str) -> None:
        """"""
        cells: dict = self.cells[vt_symbol]

        params: dict = {}
        params["price_spread"] = cells["price_spread"].get_value()
        params["volatility_spread"] = cells["volatility_spread"].get_value()

        self.algo_engine.start_algo_pricing(vt_symbol, params)

    def stop_algo_pricing(self, vt_symbol: str) -> None:
        """"""
        self.algo_engine.stop_algo_pricing(vt_symbol)

    def start_algo_trading(self, vt_symbol: str) -> None:
        """"""
        cells: dict = self.cells[vt_symbol]

        params = cells["direction"].get_value()
        for name in [
            "max_pos",
            "target_pos",
            "max_order_size"
        ]:
            params[name] = cells[name].get_value()

        self.algo_engine.start_algo_trading(vt_symbol, params)

    def stop_algo_trading(self, vt_symbol: str) -> None:
        """"""
        self.algo_engine.stop_algo_trading(vt_symbol)


class ElectronicEyeManager(QtWidgets.QWidget):
    """"""

    signal_log = QtCore.Signal(Event)

    def __init__(self, option_engine: OptionEngine, portfolio_name: str) -> None:
        """"""
        super().__init__()

        self.option_engine: OptionEngine = option_engine
        self.event_Engine: EventEngine = option_engine.event_engine
        self.algo_engine: OptionAlgoEngine = option_engine.algo_engine
        self.portfolio_name: str = portfolio_name

        self.init_ui()
        self.register_event()

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle("期权电子眼")

        self.algo_monitor: ElectronicEyeMonitor = ElectronicEyeMonitor(self.option_engine, self.portfolio_name)

        self.log_monitor: QtWidgets.QTextEdit = QtWidgets.QTextEdit()
        self.log_monitor.setReadOnly(True)
        self.log_monitor.setMaximumWidth(400)

        stop_pricing_button: QtWidgets.QPushButton = QtWidgets.QPushButton("停止定价")
        stop_pricing_button.clicked.connect(self.stop_pricing_for_all)

        stop_trading_button: QtWidgets.QPushButton = QtWidgets.QPushButton("停止交易")
        stop_trading_button.clicked.connect(self.stop_trading_for_all)

        self.price_spread_spin: AlgoDoubleSpinBox = AlgoDoubleSpinBox()
        self.volatility_spread_spin: AlgoDoubleSpinBox = AlgoDoubleSpinBox()
        self.direction_combo: AlgoDirectionCombo = AlgoDirectionCombo()
        self.max_order_size_spin: AlgoPositiveSpinBox = AlgoPositiveSpinBox()
        self.target_pos_spin: AlgoSpinBox = AlgoSpinBox()
        self.max_pos_spin: AlgoPositiveSpinBox = AlgoPositiveSpinBox()

        price_spread_button: QtWidgets.QPushButton = QtWidgets.QPushButton("设置")
        price_spread_button.clicked.connect(self.set_price_spread_for_all)

        volatility_spread_button: QtWidgets.QPushButton = QtWidgets.QPushButton("设置")
        volatility_spread_button.clicked.connect(self.set_volatility_spread_for_all)

        direction_button: QtWidgets.QPushButton = QtWidgets.QPushButton("设置")
        direction_button.clicked.connect(self.set_direction_for_all)

        max_order_size_button: QtWidgets.QPushButton = QtWidgets.QPushButton("设置")
        max_order_size_button.clicked.connect(self.set_max_order_size_for_all)

        target_pos_button: QtWidgets.QPushButton = QtWidgets.QPushButton("设置")
        target_pos_button.clicked.connect(self.set_target_pos_for_all)

        max_pos_button: QtWidgets.QPushButton = QtWidgets.QPushButton("设置")
        max_pos_button.clicked.connect(self.set_max_pos_for_all)

        QLabel = QtWidgets.QLabel
        grid: QtWidgets.QGridLayout = QtWidgets.QGridLayout()
        grid.addWidget(QLabel("价格价差"), 0, 0)
        grid.addWidget(self.price_spread_spin, 0, 1)
        grid.addWidget(price_spread_button, 0, 2)
        grid.addWidget(QLabel("隐波价差"), 1, 0)
        grid.addWidget(self.volatility_spread_spin, 1, 1)
        grid.addWidget(volatility_spread_button, 1, 2)
        grid.addWidget(QLabel("持仓范围"), 2, 0)
        grid.addWidget(self.max_pos_spin, 2, 1)
        grid.addWidget(max_pos_button, 2, 2)
        grid.addWidget(QLabel("目标持仓"), 3, 0)
        grid.addWidget(self.target_pos_spin, 3, 1)
        grid.addWidget(target_pos_button, 3, 2)
        grid.addWidget(QLabel("最大委托"), 4, 0)
        grid.addWidget(self.max_order_size_spin, 4, 1)
        grid.addWidget(max_order_size_button, 4, 2)
        grid.addWidget(QLabel("方向"), 5, 0)
        grid.addWidget(self.direction_combo, 5, 1)
        grid.addWidget(direction_button, 5, 2)

        hbox1: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        hbox1.addWidget(stop_pricing_button)
        hbox1.addWidget(stop_trading_button)

        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(grid)
        vbox.addWidget(self.log_monitor)

        hbox: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.algo_monitor)
        hbox.addLayout(vbox)

        self.setLayout(hbox)

    def register_event(self) -> None:
        """"""
        self.signal_log.connect(self.process_log_event)

        self.event_Engine.register(EVENT_OPTION_ALGO_LOG, self.signal_log.emit)

    def process_log_event(self, event: Event) -> None:
        """"""
        log: LogData = event.data
        timestr: str = log.time.strftime("%H:%M:%S")
        msg: str = f"{timestr}  {log.msg}"
        self.log_monitor.append(msg)

    def show(self) -> None:
        """"""
        self.algo_engine.init_engine(self.portfolio_name)
        self.algo_monitor.resizeColumnsToContents()
        super().showMaximized()

    def set_price_spread_for_all(self) -> None:
        """"""
        price_spread: float = self.price_spread_spin.get_value()

        for cells in self.algo_monitor.cells.values():
            if cells["price_spread"].isEnabled():
                cells["price_spread"].setValue(price_spread)

    def set_volatility_spread_for_all(self) -> None:
        """"""
        volatility_spread: float = self.volatility_spread_spin.get_value()

        for cells in self.algo_monitor.cells.values():
            if cells["volatility_spread"].isEnabled():
                cells["volatility_spread"].setValue(volatility_spread)

    def set_direction_for_all(self) -> None:
        """"""
        ix: int = self.direction_combo.currentIndex()

        for cells in self.algo_monitor.cells.values():
            if cells["direction"].isEnabled():
                cells["direction"].setCurrentIndex(ix)

    def set_max_order_size_for_all(self) -> None:
        """"""
        size: int = self.max_order_size_spin.get_value()

        for cells in self.algo_monitor.cells.values():
            if cells["max_order_size"].isEnabled():
                cells["max_order_size"].setValue(size)

    def set_target_pos_for_all(self) -> None:
        """"""
        pos: int = self.target_pos_spin.get_value()

        for cells in self.algo_monitor.cells.values():
            if cells["target_pos"].isEnabled():
                cells["target_pos"].setValue(pos)

    def set_max_pos_for_all(self) -> None:
        """"""
        pos: int = self.max_pos_spin.get_value()

        for cells in self.algo_monitor.cells.values():
            if cells["max_pos"].isEnabled():
                cells["max_pos"].setValue(pos)

    def stop_pricing_for_all(self) -> None:
        """"""
        for vt_symbol in self.algo_monitor.cells.keys():
            self.algo_monitor.stop_algo_pricing(vt_symbol)

    def stop_trading_for_all(self) -> None:
        """"""
        for vt_symbol in self.algo_monitor.cells.keys():
            self.algo_monitor.stop_algo_trading(vt_symbol)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """"""
        self.algo_monitor.save_setting()
        event.accept()


class VolatilityDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """"""

    def __init__(self) -> None:
        """"""
        super().__init__()

        self.setDecimals(1)
        self.setSuffix("%")
        self.setMaximum(200.0)
        self.setMinimum(0)

    def get_value(self) -> float:
        """"""
        return self.value()


class PricingVolatilityManager(QtWidgets.QWidget):
    """"""

    signal_timer: QtCore.Signal = QtCore.Signal(Event)

    def __init__(self, option_engine: OptionEngine, portfolio_name: str) -> None:
        """"""
        super().__init__()

        self.option_engine: OptionEngine = option_engine
        self.event_engine: EventEngine = option_engine.event_engine
        self.portfolio: PortfolioData = option_engine.get_portfolio(portfolio_name)

        self.cells: Dict[Tuple, Dict] = {}
        self.chain_symbols: List[str] = []
        self.chain_atm_index: Dict[str, str] = {}

        self.init_ui()
        self.register_event()

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle("波动率管理")

        tab: QtWidgets.QTabWidget = QtWidgets.QTabWidget()
        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addWidget(tab)
        self.setLayout(vbox)

        self.chain_symbols: list = list(self.portfolio.chains.keys())
        self.chain_symbols.sort()

        for chain_symbol in self.chain_symbols:
            chain: ChainData = self.portfolio.get_chain(chain_symbol)

            table: QtWidgets.QTableWidget = QtWidgets.QTableWidget()
            table.setEditTriggers(table.NoEditTriggers)
            table.verticalHeader().setVisible(False)
            table.setRowCount(len(chain.indexes))
            table.horizontalHeader().setSectionResizeMode(
                QtWidgets.QHeaderView.Stretch
            )

            labels: list = [
                "行权价",
                "OTM隐波",
                "CALL隐波",
                "PUT隐波",
                "定价隐波",
                "执行拟合"
            ]
            table.setColumnCount(len(labels))
            table.setHorizontalHeaderLabels(labels)

            for row, index in enumerate(chain.indexes):
                index_cell: IndexCell = IndexCell(index)
                otm_impv_cell: MonitorCell = MonitorCell("")
                call_impv_cell: MonitorCell = MonitorCell("")
                put_impv_cell: MonitorCell = MonitorCell("")

                set_func = partial(
                    self.set_pricing_impv,
                    chain_symbol=chain_symbol,
                    index=index
                )
                pricing_impv_spin: VolatilityDoubleSpinBox = VolatilityDoubleSpinBox()
                pricing_impv_spin.setAlignment(QtCore.Qt.AlignCenter)
                pricing_impv_spin.valueChanged.connect(set_func)

                check: QtWidgets.QCheckBox = QtWidgets.QCheckBox()

                check_hbox: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
                check_hbox.setAlignment(QtCore.Qt.AlignCenter)
                check_hbox.addWidget(check)

                check_widget: QtWidgets.QWidget = QtWidgets.QWidget()
                check_widget.setLayout(check_hbox)

                table.setItem(row, 0, index_cell)
                table.setItem(row, 1, otm_impv_cell)
                table.setItem(row, 2, call_impv_cell)
                table.setItem(row, 3, put_impv_cell)
                table.setCellWidget(row, 4, pricing_impv_spin)
                table.setCellWidget(row, 5, check_widget)

                cells: dict = {
                    "otm_impv": otm_impv_cell,
                    "call_impv": call_impv_cell,
                    "put_impv": put_impv_cell,
                    "pricing_impv": pricing_impv_spin,
                    "check": check
                }

                self.cells[(chain_symbol, index)] = cells

            reset_func = partial(self.reset_pricing_impv, chain_symbol=chain_symbol)
            button_reset: QtWidgets.QPushButton = QtWidgets.QPushButton("重置")
            button_reset.clicked.connect(reset_func)

            fit_func = partial(self.fit_pricing_impv, chain_symbol=chain_symbol)
            button_fit: QtWidgets.QPushButton = QtWidgets.QPushButton("拟合")
            button_fit.clicked.connect(fit_func)

            increase_func = partial(self.increase_pricing_impv, chain_symbol=chain_symbol)
            button_increase: QtWidgets.QPushButton = QtWidgets.QPushButton("+0.1%")
            button_increase.clicked.connect(increase_func)

            decrease_func = partial(self.decrease_pricing_impv, chain_symbol=chain_symbol)
            button_decrease: QtWidgets.QPushButton = QtWidgets.QPushButton("-0.1%")
            button_decrease.clicked.connect(decrease_func)

            hbox: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
            hbox.addWidget(button_reset)
            hbox.addWidget(button_fit)
            hbox.addWidget(button_increase)
            hbox.addWidget(button_decrease)

            vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
            vbox.addLayout(hbox)
            vbox.addWidget(table)

            chain_widget: QtWidgets.QWidget = QtWidgets.QWidget()
            chain_widget.setLayout(vbox)
            tab.addTab(chain_widget, chain_symbol)

            self.update_pricing_impv(chain_symbol)

            self.default_foreground = otm_impv_cell.foreground()
            self.default_background = otm_impv_cell.background()

            table.resizeRowsToContents()

    def register_event(self) -> None:
        """"""
        self.signal_timer.connect(self.process_timer_event)

        self.event_engine.register(EVENT_TIMER, self.signal_timer.emit)

    def process_timer_event(self, event: Event) -> None:
        """"""
        for chain_symbol in self.chain_symbols:
            self.update_chain_impv(chain_symbol)

    def reset_pricing_impv(self, chain_symbol: str) -> None:
        """
        Set pricing impv to the otm mid impv of each strike price.
        """
        chain: ChainData = self.portfolio.get_chain(chain_symbol)
        atm_index: str = chain.atm_index

        for index in chain.indexes:
            call: OptionData = chain.calls[index]
            put: OptionData = chain.puts[index]

            if index >= atm_index:
                otm: OptionData = call
            else:
                otm: OptionData = put

            call.pricing_impv = otm.mid_impv
            put.pricing_impv = otm.mid_impv

        self.update_pricing_impv(chain_symbol)

    def fit_pricing_impv(self, chain_symbol: str) -> None:
        """
        Fit pricing impv with cubic spline algo.
        """
        chain: ChainData = self.portfolio.get_chain(chain_symbol)
        atm_index: str = chain.atm_index

        strike_prices: list = []
        pricing_impvs: list = []

        for index in chain.indexes:
            call: OptionData = chain.calls[index]
            put: OptionData = chain.puts[index]
            cells: dict = self.cells[(chain_symbol, index)]

            if not cells["check"].isChecked():
                if index >= atm_index:
                    otm: OptionData = call
                else:
                    otm: OptionData = put

                strike_prices.append(otm.strike_price)
                pricing_impvs.append(otm.pricing_impv)

        cs: interpolate.CubicSpline = interpolate.CubicSpline(strike_prices, pricing_impvs)

        for index in chain.indexes:
            call: OptionData = chain.calls[index]
            put: OptionData = chain.puts[index]

            new_impv: float = float(cs(call.strike_price))
            call.pricing_impv = new_impv
            put.pricing_impv = new_impv

        self.update_pricing_impv(chain_symbol)

    def increase_pricing_impv(self, chain_symbol: str) -> None:
        """
        Increase pricing impv of all options within a chain by 0.1%.
        """
        chain: ChainData = self.portfolio.get_chain(chain_symbol)

        for option in chain.options.values():
            option.pricing_impv += 0.001

        self.update_pricing_impv(chain_symbol)

    def decrease_pricing_impv(self, chain_symbol: str) -> None:
        """
        Decrease pricing impv of all options within a chain by 0.1%.
        """
        chain: ChainData = self.portfolio.get_chain(chain_symbol)

        for option in chain.options.values():
            option.pricing_impv -= 0.001

        self.update_pricing_impv(chain_symbol)

    def set_pricing_impv(self, value: float, chain_symbol: str, index: str) -> None:
        """"""
        new_impv: float = value / 100

        chain: ChainData = self.portfolio.get_chain(chain_symbol)

        call: OptionData = chain.calls[index]
        call.pricing_impv = new_impv

        put: OptionData = chain.puts[index]
        put.pricing_impv = new_impv

    def update_pricing_impv(self, chain_symbol: str) -> None:
        """"""
        chain: ChainData = self.portfolio.get_chain(chain_symbol)
        atm_index: str = chain.atm_index

        for index in chain.indexes:
            if index >= atm_index:
                otm: OptionData = chain.calls[index]
            else:
                otm: OptionData = chain.puts[index]

            value: int = round(otm.pricing_impv * 100, 1)

            key: tuple = (chain_symbol, index)
            cells: Optional[dict] = self.cells.get(key, None)
            if cells:
                cells["pricing_impv"].setValue(value)

    def update_chain_impv(self, chain_symbol: str) -> None:
        """"""
        chain: ChainData = self.portfolio.get_chain(chain_symbol)
        atm_index: str = chain.atm_index

        for index in chain.indexes:
            call: OptionData = chain.calls[index]
            put: OptionData = chain.puts[index]
            if index >= atm_index:
                otm: OptionData = call
            else:
                otm: OptionData = put

            cells: dict = self.cells[(chain_symbol, index)]
            cells["otm_impv"].setText(f"{otm.mid_impv:.1%}")
            cells["call_impv"].setText(f"{call.mid_impv:.1%}")
            cells["put_impv"].setText(f"{put.mid_impv:.1%}")

        current_atm_index: str = self.chain_atm_index.get(chain_symbol, "")
        if current_atm_index == atm_index:
            return
        self.chain_atm_index[chain_symbol] = atm_index

        if current_atm_index:
            old_cells: dict = self.cells[(chain_symbol, current_atm_index)]

            for field in ["otm_impv", "call_impv", "put_impv"]:
                old_cells[field].setForeground(COLOR_WHITE)
                old_cells[field].setBackground(self.default_background)

        if atm_index:
            new_cells: dict = self.cells[(chain_symbol, atm_index)]

            for field in ["otm_impv", "call_impv", "put_impv"]:
                new_cells[field].setForeground(COLOR_BLACK)
                new_cells[field].setBackground(COLOR_WHITE)
