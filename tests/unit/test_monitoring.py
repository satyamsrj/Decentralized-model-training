"""Tests for SwarmMonitor (metrics logging and display)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from p2p_ml_swarm.monitoring import SwarmMonitor


@pytest.fixture
def monitor(tmp_path):
    return SwarmMonitor(run_id="test_run", tensorboard_dir=None, enable_rich=False)


class TestLogStep:
    def test_log_step_increments_count(self, monitor):
        monitor.log_step(epoch=0, batch_id=0, loss=0.5, lr=1e-4, peer_count=2, batch_size=32)
        assert monitor.step_count == 1

    def test_multiple_steps(self, monitor):
        for i in range(20):
            monitor.log_step(epoch=0, batch_id=i, loss=0.5 - i * 0.01, lr=1e-4, peer_count=3, batch_size=32)
        assert monitor.step_count == 20


class TestAvgLoss:
    def test_avg_recent_loss(self, monitor):
        losses = [0.5, 0.4, 0.3, 0.2, 0.1]
        for i, loss in enumerate(losses):
            monitor.log_step(epoch=0, batch_id=i, loss=loss, lr=1e-4, peer_count=1, batch_size=32)
        avg = monitor.avg_recent_loss
        assert abs(avg - 0.3) < 1e-6

    def test_avg_loss_empty(self, monitor):
        assert monitor.avg_recent_loss == 0.0


class TestEpochSummary:
    def test_log_epoch_summary_no_error(self, monitor):
        # Should not raise
        monitor.log_epoch_summary(epoch=0, avg_loss=0.25, elapsed_sec=42.0)


class TestStatusTable:
    def test_print_status_table_rich_disabled(self, monitor):
        # Should not raise even when rich is disabled
        monitor.print_status_table()


class TestClose:
    def test_close_no_error(self, monitor):
        monitor.close()
