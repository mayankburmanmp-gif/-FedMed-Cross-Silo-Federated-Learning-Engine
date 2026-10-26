"""Hospital node (Flower client).

WEEK 1: scaffolding - one process per hospital, each with its own status port.
WEEK 2: local training on the hospital's private shard; only weights leave the node.

Note: Flower clients DIAL the server (the server owns the single gRPC port), so the
"separate local ports" are each node's own status endpoint: http://127.0.0.1:8081+id/
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import flwr as fl
import torch

import config as cfg
from data import get_datalist, make_loader, partition, split_datalist
from engine import evaluate, get_device, make_loss, make_optimizer, make_scaler, train_one_epoch
from model import build_model, get_weights, set_weights


def start_status_server(port, state):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(state).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


class HospitalClient(fl.client.NumPyClient):
    def __init__(self, node_id, train_loader, val_loader, device, state):
        self.node_id, self.device, self.state = node_id, device, state
        self.train_loader, self.val_loader = train_loader, val_loader
        self.model = build_model().to(device)
        self.loss_fn = make_loss()

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)  # global weights broadcast by the server
        epochs, lr = int(config["local_epochs"]), float(config["lr"])
        self.state.update(round=int(config["server_round"]), status="training")
        # Optimizer is re-created every round (stateless client), the standard FedAvg setup.
        opt, scaler = make_optimizer(self.model, lr), make_scaler(self.device)
        loss = 0.0
        for _ in range(epochs):
            loss = train_one_epoch(self.model, self.train_loader, opt, self.loss_fn, self.device, scaler)
        self.state.update(status="idle", last_train_loss=loss)
        # Only weights + a scalar metric are returned - never images or labels.
        return get_weights(self.model), len(self.train_loader.dataset), {"train_loss": float(loss)}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        self.state.update(status="evaluating")
        m = evaluate(self.model, self.val_loader, self.device)
        self.state.update(status="idle", last_val_dice=m["dice_mean"])
        return float(m["loss"]), len(self.val_loader.dataset), {k: float(v) for k, v in m.items() if k != "loss"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-id", type=int, required=True, choices=range(cfg.NUM_NODES))
    ap.add_argument("--server-address", default=cfg.SERVER_ADDRESS)
    ap.add_argument("--insecure", action="store_true", help="disable TLS (debugging only)")
    args = ap.parse_args()

    torch.manual_seed(cfg.SEED + args.node_id)
    train_all, val_all = split_datalist(get_datalist())
    train_items = partition(train_all, cfg.NUM_NODES, args.node_id)
    val_items = partition(val_all, cfg.NUM_NODES, args.node_id)

    state = {"node_id": args.node_id, "status": "idle", "round": 0,
             "train_samples": len(train_items), "val_samples": len(val_items)}
    port = cfg.STATUS_BASE_PORT + args.node_id
    start_status_server(port, state)
    print(f"[hospital-{args.node_id}] train={len(train_items)} val={len(val_items)} status=:{port}")

    client = HospitalClient(args.node_id, make_loader(train_items, True),
                            make_loader(val_items, False), get_device(), state)
    root_certs = None if args.insecure else (cfg.CERT_DIR / "ca.crt").read_bytes()
    fl.client.start_client(server_address=args.server_address, client=client.to_client(),
                           root_certificates=root_certs)


if __name__ == "__main__":
    main()
