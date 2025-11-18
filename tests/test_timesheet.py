"""Tests for freelancer time tracking."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from og.timesheet import FreelancerTimeTracker, TimeEntry, Invoice


def test_time_entry_creation():
    """Test TimeEntry creation."""
    now = datetime.now()
    entry = TimeEntry(
        start=now,
        end=now + timedelta(hours=2),
        duration_hours=2.0,
        project='ProjectX',
        client='ClientA',
        description='Development work',
        hourly_rate=Decimal('100.00'),
    )

    assert entry.duration_hours == 2.0
    assert entry.hourly_rate == Decimal('100.00')


def test_invoice_creation():
    """Test Invoice creation."""
    invoice = Invoice(
        invoice_number='INV-001',
        client='ClientA',
        issue_date=datetime.now(),
        due_date=datetime.now() + timedelta(days=30),
        entries=[],
        subtotal=Decimal('0'),
    )

    assert invoice.invoice_number == 'INV-001'


def test_invoice_calculation():
    """Test invoice total calculation."""
    now = datetime.now()
    entry = TimeEntry(
        start=now,
        end=now + timedelta(hours=5),
        duration_hours=5.0,
        project='ProjectX',
        client='ClientA',
        description='Work',
        hourly_rate=Decimal('100.00'),
        billable=True,
    )

    invoice = Invoice(
        invoice_number='INV-001',
        client='ClientA',
        issue_date=datetime.now(),
        due_date=datetime.now() + timedelta(days=30),
        entries=[entry],
        subtotal=Decimal('0'),
        tax_rate=Decimal('0.10'),
    )

    invoice.calculate_total()

    assert invoice.subtotal == Decimal('500.00')
    assert invoice.total == Decimal('550.00')  # With 10% tax


def test_tracker_creation():
    """Test FreelancerTimeTracker creation."""
    tracker = FreelancerTimeTracker()

    assert tracker is not None
    assert len(tracker._project_mapping) == 0
