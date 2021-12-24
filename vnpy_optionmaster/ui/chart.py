from typing import Dict, List

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui

from vnpy.trader.ui import QtWidgets, QtCore
from vnpy.trader.event import EVENT_TIMER

from ..base import PortfolioData
from ..engine import OptionEngine, Event
from ..time import ANNUAL_DAYS

import numpy as np
from pylab import mpl                       # noqa

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']   # set font for Chinese
mpl.rcParams['axes.unicode_minus'] = False


class OptionVolatilityChart(QtWidgets.QWidget):

    signal_timer = QtCore.pyqtSignal(Event)

    def __init__(self, option_engine: OptionEngine, portfolio_name: str):
        """"""
        super().__init__()

        self.option_engine = option_engine
        self.event_engine = option_engine.event_engine
        self.portfolio_name = portfolio_name

        self.timer_count = 0
        self.timer_trigger = 3

        self.chain_checks: Dict[str, QtWidgets.QCheckBox] = {}
        self.put_curves: Dict[str, pg.PlotCurveItem] = {}
        self.call_curves: Dict[str, pg.PlotCurveItem] = {}
        self.pricing_curves: Dict[str, pg.PlotCurveItem] = {}

        self.colors: List = [
            (255, 0, 0),
            (255, 255, 0),
            (0, 255, 0),
            (0, 0, 255),
            (0, 128, 0),
            (19, 234, 201),
            (195, 46, 212),
            (250, 194, 5),
            (0, 114, 189),
        ]

        self.init_ui()
        self.register_event()

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle("波动率曲线")

        # Create checkbox for each chain
        hbox = QtWidgets.QHBoxLayout()
        portfolio = self.option_engine.get_portfolio(self.portfolio_name)

        chain_symbols = list(portfolio.chains.keys())
        chain_symbols.sort()

        hbox.addStretch()

        for chain_symbol in chain_symbols:
            chain_check = QtWidgets.QCheckBox()
            chain_check.setText(chain_symbol.split(".")[0])
            chain_check.setChecked(True)
            chain_check.stateChanged.connect(self.update_curve_visible)

            hbox.addWidget(chain_check)
            self.chain_checks[chain_symbol] = chain_check

        hbox.addStretch()

        # Create graphics window
        pg.setConfigOptions(antialias=True)

        graphics_window = pg.GraphicsLayoutWidget()
        self.impv_chart = graphics_window.addPlot(title="隐含波动率曲线")
        self.impv_chart.showGrid(x=True, y=True)
        self.impv_chart.setLabel("left", "波动率")
        self.impv_chart.setLabel("bottom", "行权价")
        self.impv_chart.addLegend()
        self.impv_chart.setMenuEnabled(False)
        self.impv_chart.setMouseEnabled(False, False)

        for chain_symbol in chain_symbols:
            self.add_impv_curve(chain_symbol)

        # Set Layout
        vbox = QtWidgets.QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(graphics_window)
        self.setLayout(vbox)

    def register_event(self) -> None:
        """"""
        self.signal_timer.connect(self.process_timer_event)

        self.event_engine.register(EVENT_TIMER, self.signal_timer.emit)

    def process_timer_event(self, event: Event) -> None:
        """"""
        self.timer_count += 1
        if self.timer_count < self.timer_trigger:
            return
        self.timer_trigger = 0

        self.update_curve_data()

    def add_impv_curve(self, chain_symbol: str) -> None:
        """"""
        symbol_size = 14
        symbol = chain_symbol.split(".")[0]
        color = self.colors.pop(0)
        pen = pg.mkPen(color, width=2)

        self.call_curves[chain_symbol] = self.impv_chart.plot(
            symbolSize=symbol_size,
            symbol="t1",
            name=symbol + " 看涨",
            pen=pen,
            symbolBrush=color
        )
        self.put_curves[chain_symbol] = self.impv_chart.plot(
            symbolSize=symbol_size,
            symbol="t",
            name=symbol + " 看跌",
            pen=pen,
            symbolBrush=color
        )
        self.pricing_curves[chain_symbol] = self.impv_chart.plot(
            symbolSize=symbol_size,
            symbol="o",
            name=symbol + " 定价",
            pen=pen,
            symbolBrush=color
        )

    def update_curve_data(self) -> None:
        """"""
        portfolio: PortfolioData = self.option_engine.get_portfolio(self.portfolio_name)

        for chain in portfolio.chains.values():
            call_impv = []
            put_impv = []
            pricing_impv = []
            strike_prices = []

            for index in chain.indexes:
                call = chain.calls[index]
                call_impv.append(call.mid_impv * 100)
                pricing_impv.append(call.pricing_impv * 100)
                strike_prices.append(call.strike_price)

                put = chain.puts[index]
                put_impv.append(put.mid_impv * 100)

            self.call_curves[chain.chain_symbol].setData(
                y=call_impv,
                x=strike_prices
            )
            self.put_curves[chain.chain_symbol].setData(
                y=put_impv,
                x=strike_prices
            )
            self.pricing_curves[chain.chain_symbol].setData(
                y=pricing_impv,
                x=strike_prices
            )

    def update_curve_visible(self) -> None:
        """"""
        self.impv_chart.clear()

        for chain_symbol, checkbox in self.chain_checks.items():
            if checkbox.isChecked():
                call_curve = self.call_curves[chain_symbol]
                put_curve = self.put_curves[chain_symbol]
                pricing_curve = self.pricing_curves[chain_symbol]

                self.impv_chart.addItem(call_curve)
                self.impv_chart.addItem(put_curve)
                self.impv_chart.addItem(pricing_curve)


class ScenarioAnalysisChart(QtWidgets.QWidget):
    """"""

    def __init__(self, option_engine: OptionEngine, portfolio_name: str):
        """"""
        super().__init__()

        self.option_engine = option_engine
        self.portfolio_name = portfolio_name

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle("情景分析")

        # Create widgets
        self.price_change_spin = QtWidgets.QSpinBox()
        self.price_change_spin.setSuffix("%")
        self.price_change_spin.setMinimum(2)
        self.price_change_spin.setValue(10)

        self.impv_change_spin = QtWidgets.QSpinBox()
        self.impv_change_spin.setSuffix("%")
        self.impv_change_spin.setMinimum(2)
        self.impv_change_spin.setValue(10)

        self.time_change_spin = QtWidgets.QSpinBox()
        self.time_change_spin.setSuffix("日")
        self.time_change_spin.setMinimum(0)
        self.time_change_spin.setValue(1)

        self.target_combo = QtWidgets.QComboBox()
        self.target_combo.addItems([
            "盈亏",
            "Delta",
            "Gamma",
            "Theta",
            "Vega"
        ])

        button = QtWidgets.QPushButton("执行分析")
        button.clicked.connect(self.run_analysis)

        self.w = gl.GLViewWidget()
        self.w.setMinimumHeight(500)
        self.w.setMinimumWidth(500)

        # Set layout
        hbox1 = QtWidgets.QHBoxLayout()
        hbox1.addWidget(QtWidgets.QLabel("目标数据"))
        hbox1.addWidget(self.target_combo)
        hbox1.addWidget(QtWidgets.QLabel("时间衰减"))
        hbox1.addWidget(self.time_change_spin)
        hbox1.addStretch()

        hbox2 = QtWidgets.QHBoxLayout()
        hbox2.addWidget(QtWidgets.QLabel("价格变动"))
        hbox2.addWidget(self.price_change_spin)
        hbox2.addWidget(QtWidgets.QLabel("波动率变动"))
        hbox2.addWidget(self.impv_change_spin)
        hbox2.addStretch()
        hbox2.addWidget(button)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addWidget(self.w)

        self.setLayout(vbox)

    def run_analysis(self) -> None:
        """"""
        # Generate range
        portfolio = self.option_engine.get_portfolio(self.portfolio_name)

        price_change_range = self.price_change_spin.value()
        price_changes = np.arange(-price_change_range, price_change_range + 1) / 100

        impv_change_range = self.impv_change_spin.value()
        impv_changes = np.arange(-impv_change_range, impv_change_range + 1) / 100

        time_change = self.time_change_spin.value() / ANNUAL_DAYS
        target_name = self.target_combo.currentText()

        # Check underlying price exists
        for underlying in portfolio.underlyings.values():
            if not underlying.mid_price:
                QtWidgets.QMessageBox.warning(
                    self,
                    "无法执行情景分析",
                    f"标的物{underlying.symbol}当前中间价为{underlying.mid_price}",
                    QtWidgets.QMessageBox.Ok
                )
                return

        # Run analysis calculation
        pnls = []
        deltas = []
        gammas = []
        thetas = []
        vegas = []

        for impv_change in impv_changes:
            pnl_buf = []
            delta_buf = []
            gamma_buf = []
            theta_buf = []
            vega_buf = []

            for price_change in price_changes:
                portfolio_pnl = 0
                portfolio_delta = 0
                portfolio_gamma = 0
                portfolio_theta = 0
                portfolio_vega = 0

                # Calculate underlying pnl
                for underlying in portfolio.underlyings.values():
                    if not underlying.net_pos:
                        continue

                    value = underlying.mid_price * underlying.net_pos * underlying.size
                    portfolio_pnl += value * price_change
                    portfolio_delta += value / 100

                # Calculate option pnl
                for option in portfolio.options.values():
                    if not option.net_pos:
                        continue

                    new_underlying_price = option.underlying.mid_price * (1 + price_change)
                    new_time_to_expiry = max(option.time_to_expiry - time_change, 0)
                    new_mid_impv = option.mid_impv * (1 + impv_change)

                    new_price, delta, gamma, theta, vega = option.calculate_greeks(
                        new_underlying_price,
                        option.strike_price,
                        option.interest_rate,
                        new_time_to_expiry,
                        new_mid_impv,
                        option.option_type
                    )

                    diff = new_price - option.tick.last_price
                    multiplier = option.net_pos * option.size

                    portfolio_pnl += diff * multiplier
                    portfolio_delta += delta * multiplier
                    portfolio_gamma += gamma * multiplier
                    portfolio_theta += theta * multiplier
                    portfolio_vega += vega * multiplier

                pnl_buf.append(portfolio_pnl)
                delta_buf.append(portfolio_delta)
                gamma_buf.append(portfolio_gamma)
                theta_buf.append(portfolio_theta)
                vega_buf.append(portfolio_vega)

            pnls.append(pnl_buf)
            deltas.append(delta_buf)
            gammas.append(gamma_buf)
            thetas.append(theta_buf)
            vegas.append(vega_buf)

        # Plot chart
        if target_name == "盈亏":
            target_data = pnls
        elif target_name == "Delta":
            target_data = deltas
        elif target_name == "Gamma":
            target_data = gammas
        elif target_name == "Theta":
            target_data = thetas
        else:
            target_data = vegas

        self.update_chart(price_changes * 100, impv_changes * 100, target_data, target_name)

    def update_chart(
        self,
        impv_changes: np.array,
        price_changes: np.array,
        target_data: List[List[float]],
        target_name: str
    ) -> None:

        self.w.clear()
        self.w.reset()
        self.w.show()
        self.w.setWindowTitle(target_name)
        self.w.setCameraPosition(distance=85)
        self.w.pan(dx=0, dy=0, dz=15, relative="global")

        # cal zoom percentage
        limit_price_changes = int(price_changes[0])
        zoom_x = self.zoom(limit_price_changes)
        space_x = np.linspace(-zoom_x, zoom_x, 9)

        limit_impv_changes = int(impv_changes[0])
        zoom_y = self.zoom(limit_impv_changes)
        space_y = np.linspace(-zoom_y, zoom_y, 9)

        z = np.array(target_data)
        max_z = np.max(z)
        min_z = np.min(z)
        space_z = np.linspace(min_z, max_z, 9)

        # init gril
        self.init_gril(space_x, space_y, space_z, target_name)

        # draw
        zoom = max_z - min_z
        if max_z or min_z:
            z = z / zoom * 32
        else:
            z = z * 0

        p = gl.GLSurfacePlotItem(
            x=-price_changes / zoom_x * 20,
            y=impv_changes / zoom_y * 20,
            z=z, shader="normalColor",
            computeNormals=False,
            smooth=False)
        p.translate(0, 0, -np.min(z) + 4)

        self.w.addItem(p)
        self.show()

    def init_gril(self, space_x, space_y, space_z, target_name) -> None:

        gx = gl.GLGridItem()
        gx.setSize(40, 40, 0)
        gx.setSpacing(4, 5, 0)
        gx.rotate(90, 0, 1, 0)
        gx.translate(-20, 0, 20)
        self.w.addItem(gx)

        gy = gl.GLGridItem()
        gy.setSize(40, 40, 0)
        gy.setSpacing(5, 4, 0)
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -20, 20)
        self.w.addItem(gy)

        gz = gl.GLGridItem()
        gz.setSize(40, 40, 0)
        gz.setSpacing(5, 5, 0)
        gz.translate(0, 0, 0)
        self.w.addItem(gz)

        self.add_labels_ticks(space_x, space_y, space_z, target_name)

    def add_labels_ticks(self, space_x, space_y, space_z, target_name) -> None:
        self.xpos = [(-20.0, 20.0, -2.0),
                     (-15.0, 20.0, -2.0),
                     (-10.0, 20.0, -2.0),
                     (-5.0, 20.0, -2.0),
                     (0.0, 20.0, -2.0),
                     (5.0, 20.0, -2.0),
                     (10.0, 20.0, -2.0),
                     (15.0, 20.0, -2.0),
                     (20.0, 20.0, -2.0)]
        self.xpos.reverse()

        self.ypos = [(21.0, -21.0, -2.0),
                     (21.0, -16.0, -2.0),
                     (21.0, -11.0, -2.0),
                     (21.0, -6.0, -2.0),
                     (21.0, -1.0, -2.0),
                     (21.0, 4.0, -2.0),
                     (21.0, 9.0, -2.0),
                     (21.0, 14.0, -2.0),
                     (21.0, 19.0, -2.0)]

        self.zpos = [(-21.0, 21.0, 5.0),
                     (-21.0, 21.0, 9.0),
                     (-21.0, 21.0, 13.0),
                     (-21.0, 21.0, 17.0),
                     (-21.0, 21.0, 21.0),
                     (-21.0, 21.0, 25.0),
                     (-21.0, 21.0, 29.0),
                     (-21.0, 21.0, 33.0),
                     (-21.0, 21.0, 37.0)]

        self.labels_pos = [(28.0, -1.0, -2.0),
                           (0.0, 22.0, -2.0),
                           (-21.0, 26.0, 21.0)]

        self.xtext = [str(int(i)) for i in space_x]

        self.ytext = [str(int(i)) for i in space_y]

        self.ztext = [str(int(i)) for i in space_z]

        self.labels_text = ["价格涨跌%", "波动率涨跌%", target_name]

        font = QtGui.QFont("Times", 10)

        for i in range(9):
            val = gl.GLTextItem()
            val.setData(pos=self.xpos[i], color=(255, 255, 255, 255), text=self.xtext[i], font=font)
            self.w.addItem(val)

        for i in range(9):
            val = gl.GLTextItem()
            val.setData(pos=self.ypos[i], color=(255, 255, 255, 255), text=self.ytext[i], font=font)
            self.w.addItem(val)

        for i in range(9):
            val = gl.GLTextItem()
            val.setData(pos=self.zpos[i], color=(255, 255, 255, 255), text=self.ztext[i], font=font)
            self.w.addItem(val)

        for i in range(3):
            val = gl.GLTextItem()
            val.setData(pos=self.labels_pos[i], color=(255, 255, 255, 255), text=self.labels_text[i], font=font)
            self.w.addItem(val)

    def zoom(self, length) -> int:
        length = abs(length)
        if length < 8:
            return 8
        elif length < 16:
            return 16
        elif length < 20:
            return 20
        elif length < 24:
            return 24
        elif length < 32:
            return 32
        elif length < 40:
            return 40
        elif length < 60:
            return 60
        elif length < 80:
            return 80
        else:
            return 100
