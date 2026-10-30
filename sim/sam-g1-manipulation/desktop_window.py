"""Native Qt controls with an embedded, directly rendered MuJoCo viewport."""

import json
import sys
import threading

import mujoco
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sensor_renderer import STATE
from trace_format import format_message


class MuJoCoView(QOpenGLWindow):
    def __init__(self, console):
        # NoPartialUpdate renders directly into the native window framebuffer.
        # Unlike QOpenGLWidget, no Qt FBO / MuJoCo window-buffer remapping is needed.
        super().__init__(QOpenGLWindow.UpdateBehavior.NoPartialUpdate)
        self.console = console
        self.model = self.data = self.scene = self.render_context = None
        self.camera = mujoco.MjvCamera()
        self.options = mujoco.MjvOption()
        self.last_mouse = None
        self.camera.lookat[:] = [0.45, 0, 0.7]
        self.camera.distance = 2.5
        self.camera.azimuth = 140
        self.camera.elevation = -25

    def initializeGL(self):
        self.context().aboutToBeDestroyed.connect(self.release_graphics)

    def release_graphics(self):
        self.makeCurrent()
        if self.render_context:
            self.render_context.free()
            self.render_context = None
        if self.scene:
            self.scene = None
        self.model = self.data = None
        self.doneCurrent()

    def paintGL(self):
        with self.console.view_lock:
            model, state = self.console.view_model, self.console.view_state
        if model is None:
            self.context().functions().glClear(0x00004000)  # GL_COLOR_BUFFER_BIT
            return
        if model is not self.model:
            if self.render_context:
                self.render_context.free()
            self.model = model
            self.data = mujoco.MjData(model)
            self.scene = mujoco.MjvScene(model, maxgeom=10000)
            self.render_context = mujoco.MjrContext(
                model, mujoco.mjtFontScale.mjFONTSCALE_150
            )
        # Never read the live simulation data from the GUI thread.
        mujoco.mj_setState(model, self.data, state, STATE)
        mujoco.mj_forward(model, self.data)
        mujoco.mjv_updateScene(
            model,
            self.data,
            self.options,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        scale = self.devicePixelRatio()
        viewport = mujoco.MjrRect(
            0, 0, round(self.width() * scale), round(self.height() * scale)
        )
        mujoco.mjr_render(viewport, self.scene, self.render_context)
        self.console.view_frames += 1

    def mousePressEvent(self, event):
        self.last_mouse = event.position()
        event.accept()

    def mouseMoveEvent(self, event):
        if self.last_mouse is None or self.scene is None:
            return
        position = event.position()
        dx, dy = position.x() - self.last_mouse.x(), position.y() - self.last_mouse.y()
        self.last_mouse = position
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if event.buttons() & Qt.MouseButton.RightButton:
            action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_H
                if shift
                else mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        elif event.buttons() & Qt.MouseButton.LeftButton:
            action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H
                if shift
                else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            return
        height = max(1, self.height())
        mujoco.mjv_moveCamera(
            self.model, action, dx / height, dy / height, self.scene, self.camera
        )
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        self.last_mouse = None
        event.accept()

    def wheelEvent(self, event):
        if self.scene:
            delta = event.angleDelta().y() / 120
            if not delta:
                delta = event.pixelDelta().y() / 30
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_ZOOM,
                0,
                -0.08 * delta,
                self.scene,
                self.camera,
            )
            self.update()
        event.accept()


class RobotWindow(QMainWindow):
    def __init__(self, console, show_panel=True):
        super().__init__()
        self.console = console
        self.messages_key = None
        self.setWindowTitle("Robot Lab — MuJoCo & Controls")
        self.resize(1440, 900)
        self.setMinimumSize(1000, 650)
        self.viewport = MuJoCoView(console)
        container = QWidget.createWindowContainer(self.viewport)
        container.setMinimumSize(440, 350)
        container.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(container)
        self.sidebar = QWidget()
        self.sidebar.setMinimumWidth(360)
        self.sidebar.setMaximumWidth(750)
        layout = QVBoxLayout(self.sidebar)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        self.status = QLabel("Starting simulation…")
        self.status.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(self.status)
        self.detail = QLabel(
            "Drag to orbit · right-drag to pan · scroll to zoom\nYour viewing camera is independent of the AI."
        )
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Robot actions and feedback appear here.")
        self.log.setMaximumBlockCount(1500)
        self.log.setAccessibleName("Robot action log")
        self.trace = QPlainTextEdit()
        self.trace.setReadOnly(True)
        self.trace.setMaximumBlockCount(15000)
        self.trace.setAccessibleName("Full decision and controller trace")
        tabs = QTabWidget()
        tabs.addTab(self.log, "Activity")
        tabs.addTab(self.trace, "Full debug trace")
        layout.addWidget(tabs, 1)
        self.trace_path = QLabel(
            "Logs include stated intent and measured feedback, not private model reasoning."
        )
        self.trace_path.setWordWrap(True)
        self.trace_path.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.trace_path)
        layout.addWidget(QLabel("Tell the robot what to do"))
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("Enter an instruction…")
        self.prompt.setAccessibleName("Robot command")
        self.prompt.setFixedHeight(110)
        layout.addWidget(self.prompt)
        buttons = QHBoxLayout()
        self.stop = QPushButton("Stop")
        self.stop.clicked.connect(lambda: console.interrupt())
        buttons.addWidget(self.stop)
        self.reset = QPushButton("Reset")
        self.reset.clicked.connect(lambda: console.interrupt(reset=True))
        buttons.addWidget(self.reset)
        buttons.addStretch()
        self.send = QPushButton(
            "Send  ⌘↵" if sys.platform == "darwin" else "Send  Ctrl+Enter"
        )
        self.send.setDefault(True)
        self.send.clicked.connect(self.submit)
        buttons.addWidget(self.send)
        layout.addLayout(buttons)
        shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        shortcut.activated.connect(self.submit)
        self.shortcut = shortcut
        split.addWidget(self.sidebar)
        split.setChildrenCollapsible(False)
        split.setSizes([960, 480])
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        self.sidebar.setVisible(show_panel)
        self.setCentralWidget(split)
        self.refresh = QTimer(self)
        self.refresh.timeout.connect(self.poll)
        self.refresh.start(250)
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.viewport.update)
        self.render_timer.start(33)
        self.prompt.setFocus()

    def submit(self):
        text = self.prompt.toPlainText().strip()
        if not text or not self.send.isEnabled():
            return
        try:
            if len(text) > 2000:
                raise ValueError("Enter a command of at most 2000 characters")
            self.console.prompt(text)
            self.poll()
        except (ValueError, RuntimeError) as exc:
            self.status.setText(str(exc))

    def poll(self):
        state = self.console.state()
        status = state["status"]
        self.status.setText(status)
        ready = self.console.view_model is not None and status not in ("Starting simulation", "Resetting")
        # Sending while the robot works replaces the current task (e.g. "drop it").
        self.send.setEnabled(ready and not state["fault"])
        self.send.setToolTip("Replaces the current task" if state["busy"] else "")
        self.reset.setEnabled(ready or bool(state["fault"]))
        self.stop.setEnabled(self.console.robot is not None)
        if state.get("trace_directory"):
            self.trace_path.setText("Trace files: " + state["trace_directory"])
        display = state["display"]
        self.detail.setText(
            f"Local simulation · {display['real_time_ratio']:.2f}× real time · AI camera: {display['agent_camera']}\n"
            "Drag to orbit · right-drag to pan · scroll to zoom"
        )
        messages = state["messages"]
        key = json.dumps(messages, sort_keys=True)
        if key != self.messages_key:
            self.messages_key = key
            scroll = self.log.verticalScrollBar()
            at_end = scroll.value() >= scroll.maximum() - 5
            rows = [format_message(row) for row in messages]
            previous = scroll.value()
            self.log.setPlainText("\n\n".join(rows))
            scroll.setValue(scroll.maximum() if at_end else previous)
            debug_scroll = self.trace.verticalScrollBar()
            debug_end = debug_scroll.value() >= debug_scroll.maximum() - 5
            previous_debug = debug_scroll.value()
            self.trace.setPlainText(
                "\n\n".join(format_message(row, detailed=True) for row in messages)
            )
            debug_scroll.setValue(
                debug_scroll.maximum() if debug_end else previous_debug
            )

    def closeEvent(self, event):
        self.refresh.stop()
        self.render_timer.stop()
        self.console.shutdown()
        self.viewport.release_graphics()
        event.accept()


def run_desktop(console, show_panel=True):
    # MuJoCo's fixed-function OpenGL renderer needs the compatibility profile.
    # Set this before QApplication creates any macOS contexts.
    surface = QSurfaceFormat()
    surface.setVersion(2, 1)
    surface.setProfile(QSurfaceFormat.OpenGLContextProfile.NoProfile)
    surface.setDepthBufferSize(24)
    surface.setStencilBufferSize(8)
    surface.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(surface)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Robot Lab")
    window = RobotWindow(console, show_panel)

    def physics():
        try:
            console.run()
        except (RuntimeError, ValueError, OSError) as exc:
            console.fault = str(exc)
            console.status = "Simulation stopped: " + str(exc)

    worker = threading.Thread(target=physics, name="robot-physics", daemon=True)
    worker.start()
    window.show()
    window.raise_()
    window.activateWindow()
    try:
        app.exec()
    finally:
        console.shutdown()
        worker.join(timeout=5)
