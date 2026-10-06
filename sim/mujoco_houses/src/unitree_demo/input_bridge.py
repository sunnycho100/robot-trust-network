"""Small pygame window that turns WASD/QE or a gamepad into a velocity command."""

from __future__ import annotations

import argparse
import json
import socket
import struct
import time

import numpy as np
import pygame


def deadzone(value: float, threshold: float = 0.12) -> float:
    return 0.0 if abs(value) < threshold else value


def bubble(screen, font, rect, colour, title: str, message: str) -> None:
    pygame.draw.rect(screen, colour, rect, border_radius=18)
    pygame.draw.rect(screen, (235, 238, 245), rect, width=2, border_radius=18)
    screen.blit(font.render(title, True, (255, 255, 255)), (rect.x + 16, rect.y + 12))
    screen.blit(font.render(message, True, (255, 255, 255)), (rect.x + 16, rect.y + 46))


def draw(screen, font, command: np.ndarray, gamepad: bool, status: dict, duo: bool) -> None:
    screen.fill((25, 28, 34))
    automatic = duo and status.get("stage", "").split(" · ", 1)[0].isdigit()
    if automatic:
        controls = status.get("demo_title", "AUTOMATIC MQTT STORY")
    else:
        controls = "Gamepad: left stick move, right stick turn" if gamepad else "Hold W/S move, A/D turn, Q/E lateral - release to stop"
    visible_command = status.get("leader_command", command)
    screen.blit(font.render(controls, True, (235, 238, 245)), (20, 18))
    screen.blit(font.render(f"Leader cmd: {visible_command[0]:+.2f}  {visible_command[1]:+.2f}  {visible_command[2]:+.2f}", True, (235, 238, 245)), (20, 52))

    if not duo:
        screen.blit(font.render("Release all keys = zero command    Esc = quit", True, (235, 238, 245)), (20, 100))
        pygame.display.flip()
        return

    stage = status.get("stage", "CONNECTING")
    stage_surface = font.render(stage, True, (255, 209, 112))
    screen.blit(stage_surface, stage_surface.get_rect(center=(380, 91)))
    leader_rect = pygame.Rect(20, 120, 300, 95)
    follower_rect = pygame.Rect(440, 120, 300, 95)
    bubble(screen, font, leader_rect, (145, 67, 40), "G1 LEADER", status.get("leader_text", "Ready - waiting for input"))
    bubble(screen, font, follower_rect, (35, 89, 155), status.get("follower_title", "FOLLOWER"), status.get("follower_text", "Waiting for broker message"))

    pygame.draw.line(screen, (255, 181, 71), (320, 167), (360, 167), 4)
    pygame.draw.line(screen, (87, 180, 255), (400, 167), (440, 167), 4)
    pygame.draw.polygon(screen, (255, 181, 71), [(360, 160), (374, 167), (360, 174)])
    pygame.draw.polygon(screen, (87, 180, 255), [(426, 160), (440, 167), (426, 174)])
    pygame.draw.circle(screen, (88, 92, 105), (400, 167), 35)
    broker_label = font.render("MQTT", True, (255, 255, 255))
    screen.blit(broker_label, broker_label.get_rect(center=(400, 167)))

    connected = status.get("connected", False)
    state_colour = (91, 210, 135) if connected else (245, 94, 94)
    state = "READY" if connected else "IN TRANSIT"
    pygame.draw.circle(screen, state_colour, (32, 257), 8)
    metrics = (
        f"Follower input {state}    seq={status.get('sequence', '-')}    "
        f"message age={status.get('age_ms', '-')} ms    distance={status.get('distance_m', '-')} m"
    )
    screen.blit(font.render(metrics, True, (235, 238, 245)), (50, 244))
    delay = status.get("message_delay_ms", 0)
    decision = status.get("trust_delay_ms", 0)
    screen.blit(font.render(f"A -> MQTT -> B   decision pause {decision} ms    pose lag {delay} ms", True, (180, 187, 201)), (20, 292))
    screen.blit(font.render(f"B -> MQTT -> A   {status.get('ack_text', 'waiting for signed reply')}", True, (180, 187, 201)), (20, 326))
    screen.blit(font.render(status.get("trust_text", "Trust: waiting for a signed request"), True, (180, 187, 201)), (20, 360))
    screen.blit(font.render("Esc closes the demo", True, (180, 187, 201)), (20, 394))
    pygame.display.flip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--duo", action="store_true", help="Show the leader/follower MQTT panel")
    args = parser.parse_args()

    pygame.init()
    pygame.joystick.init()
    duo = args.duo
    screen = pygame.display.set_mode((760, 430) if duo else (620, 150))
    pygame.display.set_caption(f"Unitree {args.title} command input")
    font = pygame.font.Font(None, 26)
    joystick = None
    if pygame.joystick.get_count():
        joystick = pygame.joystick.Joystick(0)
        joystick.init()

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.setblocking(False)
    destination = ("127.0.0.1", args.port)
    clock = pygame.time.Clock()
    last_command = None
    status = {}
    last_status = None
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                running = False
            if joystick is not None:
                command = np.asarray(
                    [
                        -0.7 * deadzone(joystick.get_axis(1)),
                        -0.4 * deadzone(joystick.get_axis(0)),
                        -0.8 * deadzone(joystick.get_axis(3)),
                    ],
                    dtype=np.float32,
                )
            else:
                command = np.asarray(
                    [
                        0.5 * float(keys[pygame.K_w]) - 0.35 * float(keys[pygame.K_s]),
                        0.3 * float(keys[pygame.K_q]) - 0.3 * float(keys[pygame.K_e]),
                        0.6 * float(keys[pygame.K_a]) - 0.6 * float(keys[pygame.K_d]),
                    ],
                    dtype=np.float32,
                )
            sender.sendto(struct.pack("!fff", *command), destination)
            while True:
                try:
                    status_payload, _ = sender.recvfrom(4096)
                except BlockingIOError:
                    break
                try:
                    status = json.loads(status_payload)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            command_tuple = tuple(float(x) for x in command)
            status_key = json.dumps(status, sort_keys=True)
            if command_tuple != last_command or status_key != last_status:
                draw(screen, font, command, joystick is not None, status, duo)
                last_command = command_tuple
                last_status = status_key
            clock.tick(60)
    finally:
        # Send a final explicit zero before closing.
        sender.sendto(struct.pack("!fff", 0.0, 0.0, 0.0), destination)
        sender.close()
        pygame.quit()


if __name__ == "__main__":
    main()
