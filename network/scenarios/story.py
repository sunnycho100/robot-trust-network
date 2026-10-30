"""The guided demo in the browser: map.html with Start and Pause, the story in its events panel.

Run ./demo.sh. Host side, standard library only. Start (re)builds fresh households and plays the
steps in play.DEMO one after another; Pause holds before the next step.
"""
import functools, json, os, subprocess, sys, threading, time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import run
from nl import interpret_route, room_of
from play import DEMO, KO, ROBOTS, ensure_docker, korean
from run import A, B, COMPOSE, client, events, owner, sh

PORT = 8766
GAP_S = 3.0  # reading time between steps
HERE = os.path.dirname(os.path.abspath(__file__))


class Story:
    def __init__(self):
        self.lock = threading.Lock()
        self.lines, self.step, self.running, self.paused = [], 0, False, False

    def emit(self, kind, text):
        with self.lock:
            self.lines.append({"t": time.strftime("%H:%M:%S"), "kind": kind, "text": text})

    def snapshot(self):
        with self.lock:
            return {"lines": self.lines[-80:], "step": self.step, "total": len(DEMO),
                    "running": self.running, "paused": self.paused}

    def start(self):
        with self.lock:
            if self.running:
                return
            self.lines, self.step, self.running, self.paused = [], 0, True, False
        threading.Thread(target=self.play, daemon=True).start()

    def toggle_pause(self):
        with self.lock:
            if not self.running:
                return
            self.paused = not self.paused
            paused = self.paused
        self.emit("system", "일시정지: 지금 단계가 끝나면 멈춰요" if paused else "다시 진행해요")

    def hold(self, seconds=0.0):
        """Wait out the reading time, and longer while paused."""
        end = time.monotonic() + seconds
        while time.monotonic() < end or self.paused:
            time.sleep(0.1)

    def play(self):
        try:
            self.emit("system", "집 세 채와 브로커를 새로 준비하는 중이에요 (약 20초)")
            sh(*COMPOSE, "exec", "-T", "house-1", "pkill", "-f", "client.py visit", check=False)
            run.setup()
            self.emit("system", "준비 완료. 옆집 로봇 A가 우리 집(로봇 B)을 방문해요.")
            me = A
            for i, (say, commands) in enumerate(DEMO, 1):
                self.hold()
                with self.lock:
                    self.step = i
                self.emit("story", f"[{i}/{len(DEMO)}] {say}")
                for command in commands:
                    me = self.execute(command, me)
                self.hold(GAP_S)
            self.emit("system", "데모 끝. 시작을 누르면 처음부터 다시 해요.")
        except Exception as e:  # show it on the page instead of dying silently
            self.emit("error", f"오류: {e}")
        finally:
            with self.lock:
                self.running, self.paused = False, False

    def watch(self, since, until, timeout):
        """Show what the visiting robot reports until a note contains one of `until`."""
        seen, deadline = since, time.monotonic() + timeout
        while time.monotonic() < deadline:
            new = events("house-1")[seen:]
            seen += len(new)
            for e in new:
                if e["kind"] == "visit":
                    bad = any(w in e["note"] for w in ("rejected", "DENY", "gave up"))
                    self.emit("bad" if bad else "robot", korean(e["note"]))
                    if any(u in e["note"] for u in until):
                        return
            time.sleep(0.2)

    def execute(self, command, me):
        word, *rest = command.split()
        since = len(events("house-1"))
        if word == "as":
            me = ROBOTS[rest[0]]
            self.emit("system", f"이제 {me.split('@')[0]}로 요청해요")
        elif word == "route":
            rooms = [room_of(w) for w in rest]
            if not all(rooms):
                rooms, how = interpret_route(" ".join(rest))
                self.emit("system", f"\"{' '.join(rest)}\" -> 경로 {' > '.join(KO.get(r, r) for r in rooms)} ({how})")
            sh(*COMPOSE, "exec", "-T", "house-1", "pkill", "-f", "client.py visit", check=False)
            run.background_client("visit", me, B, *rooms, "--pace", "1.5", "--seconds", "180")
            self.watch(since, ("waiting at", "arrived", "DENY"), 20)
        elif word == "approve":
            room = room_of(rest[0]) or rest[0]
            owner("house-2", "approve", me, room)
            self.emit("system", f"집주인이 {KO.get(room, room)} 출입을 승인했어요")
            self.watch(since, ("waiting at", "arrived", "rejected"), 15)
        elif word == "bad_update":
            owner("house-2", "bad_update", me)
            self.watch(since, ("rejected", "accepted"), 6)
        else:  # fake_update, replay_update
            client("route_attack", word, me, B)
            self.watch(since, ("rejected", "accepted"), 6)
        return me


STORY = Story()


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/story":
            return self.reply(STORY.snapshot())
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/start":
            STORY.start()
        elif self.path == "/api/pause":
            STORY.toggle_pause()
        else:
            return self.send_error(404)
        self.reply(STORY.snapshot())


def main():
    os.environ.update(HOST_UID=str(os.getuid()), HOST_GID=str(os.getgid()))
    ensure_docker()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), functools.partial(Handler, directory=HERE))
    url = f"http://127.0.0.1:{PORT}/map.html"
    print(f"demo page: {url}  (press Start on the page; Ctrl+C here to stop the page)")
    if "--no-open" not in sys.argv:
        subprocess.run(["open", url])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped. The households keep running; docker compose down stops them.")


if __name__ == "__main__":
    main()
