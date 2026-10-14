"""Network interface detection and USB-LAN adapter probe for Local ML Cluster.

Identifies USB network interfaces, Wi-Fi, Ethernet, and private overlay
adapters, verifying subnet alignment and link-up status for offline mode.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)

# Keywords identifying common USB-Ethernet / USB-device network adapters
USB_INTERFACE_KEYWORDS = (
    "usb",
    "rndis",
    "ecm",
    "ncm",
    "asix",
    "realtek usb",
    "ax88",
    "lan78",
    "usb-ethernet",
    "ethernet adapter",
    "remote ndis",
    "belkin",
    "anker",
    "tp-link usb",
)

# Keywords identifying virtual overlay networks (Tailscale, WireGuard, ZeroTier)
OVERLAY_INTERFACE_KEYWORDS = (
    "tailscale",
    "wireguard",
    "wg",
    "zerotier",
    "zt",
    "utun",
    "tun",
    "tap",
)


@dataclass(frozen=True)
class NetworkAddress:
    """IP address configuration for a network interface."""

    ip: str
    netmask: Optional[str] = None
    broadcast: Optional[str] = None
    is_ipv6: bool = False


@dataclass
class NetworkInterface:
    """Representation of an operating system network adapter."""

    name: str
    is_up: bool
    speed_mbps: int
    is_loopback: bool
    is_usb: bool
    is_overlay: bool
    addresses: List[NetworkAddress] = field(default_factory=list)

    @property
    def primary_ipv4(self) -> Optional[str]:
        """Return the first non-loopback IPv4 address on this interface."""
        for addr in self.addresses:
            if not addr.is_ipv6 and addr.ip != "127.0.0.1":
                return addr.ip
        return None

    def same_subnet(self, other_ip: str) -> bool:
        """Check whether another IP address belongs to the same subnet as this interface."""
        for addr in self.addresses:
            if not addr.is_ipv6 and addr.netmask:
                try:
                    network = ipaddress.IPv4Network(f"{addr.ip}/{addr.netmask}", strict=False)
                    if ipaddress.IPv4Address(other_ip) in network:
                        return True
                except ValueError:
                    continue
        return False


def probe_network_interfaces() -> List[NetworkInterface]:
    """Inspect all system network adapters and classify them.

    Distinguishes USB-to-Ethernet adapters, private overlays, Wi-Fi,
    and regular Ethernet interfaces.

    Returns:
        List of populated NetworkInterface objects.
    """
    logger.debug("Probing system network interfaces...")
    stats = psutil.net_if_stats()
    addrs = psutil.net_if_addrs()
    interfaces: List[NetworkInterface] = []

    for name, addr_list in addrs.items():
        name_lower = name.lower()
        stat = stats.get(name)
        is_up = stat.isup if stat else False
        speed = stat.speed if stat else 0

        # Classification heuristics
        is_loopback = "loopback" in name_lower or name_lower == "lo"
        is_usb = any(kw in name_lower for kw in USB_INTERFACE_KEYWORDS)
        is_overlay = any(kw in name_lower for kw in OVERLAY_INTERFACE_KEYWORDS)

        parsed_addrs: List[NetworkAddress] = []
        for saddr in addr_list:
            if saddr.family == socket.AF_INET:
                parsed_addrs.append(
                    NetworkAddress(
                        ip=saddr.address,
                        netmask=saddr.netmask,
                        broadcast=saddr.broadcast,
                        is_ipv6=False,
                    )
                )
            elif saddr.family == socket.AF_INET6:
                parsed_addrs.append(
                    NetworkAddress(
                        ip=saddr.address.split("%")[0],  # Strip scope id if present
                        netmask=saddr.netmask,
                        is_ipv6=True,
                    )
                )

        if parsed_addrs:  # Only track interfaces with assigned addresses
            if any(a.ip.startswith("127.") for a in parsed_addrs):
                is_loopback = True

        iface = NetworkInterface(
            name=name,
            is_up=is_up,
            speed_mbps=speed,
            is_loopback=is_loopback,
            is_usb=is_usb,
            is_overlay=is_overlay,
            addresses=parsed_addrs,
        )
        interfaces.append(iface)

    logger.info(
        "Network probe found %d interfaces (%d USB, %d overlay, %d up)",
        len(interfaces),
        sum(1 for i in interfaces if i.is_usb),
        sum(1 for i in interfaces if i.is_overlay),
        sum(1 for i in interfaces if i.is_up),
    )
    return interfaces


def find_usb_network_interfaces() -> List[NetworkInterface]:
    """Filter and return only active USB network adapters."""
    return [i for i in probe_network_interfaces() if i.is_usb and i.is_up]


def find_overlay_interfaces() -> List[NetworkInterface]:
    """Filter and return only active private overlay network adapters."""
    return [i for i in probe_network_interfaces() if i.is_overlay and i.is_up]
