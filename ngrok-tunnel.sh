#!/bin/bash
# Manage ngrok tunnel for RAG platform

NGROK_LOG="/tmp/ngrok.log"
NGROK_PID_FILE="/tmp/ngrok.pid"

case "$1" in
    start)
        if pgrep -f "ngrok http" > /dev/null; then
            echo "⚠️  ngrok is already running"
            exit 0
        fi

        echo "🚀 Starting ngrok tunnel..."
        nohup ngrok http 80 --log stdout > "$NGROK_LOG" 2>&1 &
        echo $! > "$NGROK_PID_FILE"

        sleep 3

        if pgrep -f "ngrok http" > /dev/null; then
            PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])" 2>/dev/null)
            echo "✅ ngrok tunnel started!"
            echo ""
            echo "Public URL: $PUBLIC_URL"
            echo "Dashboard:  http://localhost:4040"
            echo ""
            echo "⚠️  Note: Free tier shows warning page on first visit - click 'Visit Site' to continue"
        else
            echo "❌ Failed to start ngrok"
            exit 1
        fi
        ;;

    stop)
        if pgrep -f "ngrok http" > /dev/null; then
            echo "🛑 Stopping ngrok..."
            pkill -f "ngrok http"
            rm -f "$NGROK_PID_FILE"
            echo "✅ ngrok stopped"
        else
            echo "⚠️  ngrok is not running"
        fi
        ;;

    status)
        if pgrep -f "ngrok http" > /dev/null; then
            PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])" 2>/dev/null)
            echo "✅ ngrok is RUNNING"
            echo ""
            echo "Public URL: $PUBLIC_URL"
            echo "Dashboard:  http://localhost:4040"
            echo ""
            echo "Recent requests:"
            curl -s http://localhost:4040/api/requests/http | python3 -c "import sys, json; reqs=json.load(sys.stdin)['requests'][:5]; [print(f\"  {r['method']} {r['uri']} - {r['response']['status']}\") for r in reqs]" 2>/dev/null || echo "  No requests yet"
        else
            echo "❌ ngrok is NOT running"
            exit 1
        fi
        ;;

    restart)
        $0 stop
        sleep 2
        $0 start
        ;;

    logs)
        if [ -f "$NGROK_LOG" ]; then
            tail -f "$NGROK_LOG"
        else
            echo "❌ No log file found at $NGROK_LOG"
            exit 1
        fi
        ;;

    *)
        echo "Usage: $0 {start|stop|status|restart|logs}"
        echo ""
        echo "Commands:"
        echo "  start   - Start ngrok tunnel"
        echo "  stop    - Stop ngrok tunnel"
        echo "  status  - Check tunnel status and show URL"
        echo "  restart - Restart tunnel"
        echo "  logs    - Tail ngrok logs"
        exit 1
        ;;
esac
