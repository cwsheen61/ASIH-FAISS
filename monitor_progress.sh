#!/bin/bash
# Monitor scraping progress every minute

while true; do
  clear
  echo "========================================"
  echo "SCRAPING PROGRESS - $(date '+%H:%M:%S')"
  echo "========================================"
  echo ""
  echo "DATABASE STATS:"
  python scraper_trainer.py stats 2>&1 | grep -E "(Total posts:|Unique images:|Labeled posts:|ALLOWED:|DENIED:)" | head -5
  echo ""
  echo "4CHAN SCRAPER (bc70854):"
  if [ -f /tmp/claude/-home-cwsheen-Workspace-scraper-trainer/tasks/bc70854.output ]; then
    tail -3 /tmp/claude/-home-cwsheen-Workspace-scraper-trainer/tasks/bc70854.output
  else
    echo "  Not running"
  fi
  echo ""
  echo "TELEGRAM SCRAPER (bc233d8):"
  if [ -f /tmp/claude/-home-cwsheen-Workspace-scraper-trainer/tasks/bc233d8.output ]; then
    tail -3 /tmp/claude/-home-cwsheen-Workspace-scraper-trainer/tasks/bc233d8.output
  else
    echo "  Not running"
  fi
  echo ""
  echo "========================================"
  sleep 60
done
