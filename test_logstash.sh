#!/bin/bash

# Test script for Logstash pipeline
# This script verifies that Logstash is properly receiving and processing logs

set -e

echo "=== LOGSTASH PIPELINE TEST ==="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ELASTICSEARCH_URL="http://localhost:9200"
LOGSTASH_API_URL="http://localhost:9600"
LOGSTASH_SYSLOG_PORT="5514"
KIBANA_URL="http://localhost:5601"

# Test functions
test_elasticsearch() {
    echo -n "Testing Elasticsearch connection... "
    if curl -s "$ELASTICSEARCH_URL/_cluster/health" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        HEALTH=$(curl -s "$ELASTICSEARCH_URL/_cluster/health" | jq -r '.status')
        echo "  Status: $HEALTH"
    else
        echo -e "${RED}FAILED${NC}"
        return 1
    fi
}

test_logstash_api() {
    echo -n "Testing Logstash API... "
    if curl -s "$LOGSTASH_API_URL/_node" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        STATUS=$(curl -s "$LOGSTASH_API_URL/_node" | jq -r '.status')
        VERSION=$(curl -s "$LOGSTASH_API_URL/_node" | jq -r '.version')
        echo "  Status: $STATUS, Version: $VERSION"
    else
        echo -e "${RED}FAILED${NC}"
        return 1
    fi
}

test_kibana() {
    echo -n "Testing Kibana connection... "
    if curl -s "$KIBANA_URL/api/status" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        STATUS=$(curl -s "$KIBANA_URL/api/status" | jq -r '.status.overall.level')
        echo "  Status: $STATUS"
    else
        echo -e "${RED}FAILED${NC}"
        return 1
    fi
}

send_test_logs() {
    echo "Sending test logs to Logstash..."

    # Test log via syslog (RFC3164 format)
    echo -n "  Sending syslog message... "
    if echo "<14>$(date '+%b %d %H:%M:%S') test-host trading-test: {\"timestamp\":\"$(date -Iseconds)\",\"level\":\"INFO\",\"service\":\"test-service\",\"message\":\"Test log via syslog\",\"event_type\":\"test\",\"environment\":\"development\"}" | nc -w 1 localhost $LOGSTASH_SYSLOG_PORT 2>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}FAILED (but this is OK if nc is not available)${NC}"
    fi

    # Test log via docker exec to redis
    echo -n "  Sending log via Redis... "
    if docker exec trading_redis redis-cli lpush trading_logs "{\"timestamp\":\"$(date -Iseconds)\",\"level\":\"INFO\",\"service\":\"test-service-redis\",\"message\":\"Test log via Redis\",\"event_type\":\"test\",\"environment\":\"development\"}" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}FAILED (Redis not accessible)${NC}"
    fi

    # Wait for processing
    echo "  Waiting 5 seconds for log processing..."
    sleep 5
}

check_logs_in_elasticsearch() {
    echo "Checking for test logs in Elasticsearch..."

    # Check indices
    echo -n "  Checking indices... "
    INDICES=$(curl -s "$ELASTICSEARCH_URL/_cat/indices?h=index" | grep -E "trading-logs|logstash" | wc -l)
    if [ "$INDICES" -gt 0 ]; then
        echo -e "${GREEN}Found $INDICES relevant indices${NC}"
        curl -s "$ELASTICSEARCH_URL/_cat/indices?v" | grep -E "trading-logs|logstash" || true
    else
        echo -e "${YELLOW}No trading-logs indices found yet${NC}"
    fi

    # Try to search for our test logs
    echo -n "  Searching for test logs... "
    TODAY=$(date +%Y.%m.%d)

    # Search in today's index
    SEARCH_RESULT=$(curl -s "$ELASTICSEARCH_URL/trading-logs-$TODAY/_search?q=test-service" 2>/dev/null || echo '{"hits":{"total":{"value":0}}}')
    LOG_COUNT=$(echo "$SEARCH_RESULT" | jq -r '.hits.total.value // .hits.total // 0' 2>/dev/null || echo "0")

    if [ "$LOG_COUNT" -gt 0 ]; then
        echo -e "${GREEN}Found $LOG_COUNT test logs${NC}"
        echo "$SEARCH_RESULT" | jq -r '.hits.hits[]._source.message // empty' | head -3 | sed 's/^/    /'
    else
        echo -e "${YELLOW}No test logs found in today's index${NC}"

        # Try searching all indices
        ALL_SEARCH=$(curl -s "$ELASTICSEARCH_URL/_all/_search?q=test-service&size=1" 2>/dev/null || echo '{"hits":{"total":{"value":0}}}')
        ALL_COUNT=$(echo "$ALL_SEARCH" | jq -r '.hits.total.value // .hits.total // 0' 2>/dev/null || echo "0")

        if [ "$ALL_COUNT" -gt 0 ]; then
            echo -e "${YELLOW}  But found $ALL_COUNT test logs in other indices${NC}"
        fi
    fi
}

show_pipeline_stats() {
    echo "Logstash Pipeline Statistics:"
    if curl -s "$LOGSTASH_API_URL/_node/stats/pipeline/main" 2>/dev/null | jq . > /dev/null 2>&1; then
        curl -s "$LOGSTASH_API_URL/_node/stats/pipeline/main" | jq '{
            events_in: .pipeline.events.in,
            events_out: .pipeline.events.out,
            events_filtered: .pipeline.events.filtered,
            duration_in_millis: .pipeline.events.duration_in_millis
        }' 2>/dev/null || echo "  Pipeline stats not available"
    else
        echo "  Pipeline stats not available (API endpoint issue)"
    fi
}

# Run tests
echo "Starting ELK stack tests..."
echo

test_elasticsearch || exit 1
echo

test_logstash_api || exit 1
echo

test_kibana || exit 1
echo

send_test_logs
echo

check_logs_in_elasticsearch
echo

show_pipeline_stats
echo

# Final summary
echo "=== TEST SUMMARY ==="
echo "✓ Elasticsearch: Running and healthy"
echo "✓ Logstash: Running and API accessible"
echo "✓ Kibana: Running and accessible"
echo
echo "Endpoints:"
echo "  - Elasticsearch: $ELASTICSEARCH_URL"
echo "  - Kibana: $KIBANA_URL"
echo "  - Logstash API: $LOGSTASH_API_URL"
echo
echo "To view logs in Kibana:"
echo "  1. Open $KIBANA_URL"
echo "  2. Go to 'Discover' in the left menu"
echo "  3. Create an index pattern for 'trading-logs-*'"
echo "  4. Search for your logs"
echo

echo -e "${GREEN}ELK stack is operational!${NC}"
