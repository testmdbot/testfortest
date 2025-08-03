#!/usr/bin/env python3
"""
forwardbot.py - Complete Telegram Forwarding Bot (3000+ lines)
==============================================================

COMPREHENSIVE FEATURE SET:
- Three forwarding modes (with_tags, without_tags, bypass)
- Auto-forwarding rules with keyword filtering
- Premium/Admin/Owner role system
- Phone verification system
- Owner console session (stdin)
- Statistics and logging
- Rule management with inline keyboards
- Settings panel
- Export/Import functionality
- Advanced message filtering
- Scheduled forwarding
- Channel management
- User analytics
- Premium emoji detection
- Multi-language support
- Database backup system
- Rate limiting and flood control
- Custom message templates
- Webhook support
- API integration
- Advanced error handling
- Security features
- Performance monitoring
- Custom filters
- Batch operations
- Message queuing
- Real-time monitoring
- Advanced reporting
- Custom notifications
- Integration capabilities
- Extended logging
- Data visualization
- Performance optimization
"""

# bot.py (at the very top)
from dotenv import load_dotenv
load_dotenv()
from telegram.ext import CommandHandler, CallbackQueryHandler, MessageHandler, ConversationHandler, filters

# Renamed to avoid conflict with global vars if they were used elsewhere
# These are typically just integers for ConversationHandler states
BROADCAST_MESSAGE_STATE, BROADCAST_ROLE_MESSAGE_STATE = range(2)

import time as systime  # <-- standard time module
from datetime import time as dtime  # <-- datetime.time class
import asyncio
import json
import logging
import os
import re
import sys
import time
import threading
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
import uuid
from functools import wraps
import traceback
import psutil # Added for system_stats and debug_command
import shutil # For database backup

from telegram import (
    __version__ as TG_VER,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
    Bot,
    Message,
    User,
    Chat,
    CallbackQuery,
    MessageEntity,
    Contact,
    Location,
    Venue,
    Poll,
    Sticker,
    Voice,
    VideoNote,
    Document,
    InputFile, # For sending files
)
from telegram.constants import ParseMode, ChatType, MessageType
from telegram.ext import (
    AIORateLimiter,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    Defaults,
    MessageHandler,
    filters,
    JobQueue,
    Application,
    ConversationHandler,
    PersistenceInput,
)
from telegram.error import TelegramError, BadRequest, Forbidden, NetworkError

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Bot Configuration
BOT_TOKEN = os.getenv("BOT_TOKEN")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
ADMIN_IDS = list(map(int, os.getenv("ADMIN_IDS", "").split(","))) if os.getenv("ADMIN_IDS") else []
DATABASE_URL = os.getenv("DATABASE_URL", "bot_data.db") # Default to a file if not set
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8443"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "30"))

# File Paths
DATA_DIR = Path("data")
LOGS_DIR = Path("logs")
BACKUP_DIR = Path("backups")
TEMP_DIR = Path("temp")

# Create directories if they don't exist
for dir_path in [DATA_DIR, LOGS_DIR, BACKUP_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "bot_data.json" # Legacy data file
LOG_FILE = LOGS_DIR / "bot.log"
ERROR_LOG_FILE = LOGS_DIR / "errors.log"
STATS_FILE = DATA_DIR / "stats.json" # Legacy stats file
SETTINGS_FILE = DATA_DIR / "settings.json" # Legacy settings file

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Color formatter for console
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

logger = logging.getLogger(__name__)
logger.handlers = [
    logging.FileHandler(LOG_FILE, encoding='utf-8'),
    console_handler
]

# Error logger
error_logger = logging.getLogger('errors')
error_handler = logging.FileHandler(ERROR_LOG_FILE, encoding='utf-8')
error_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s\n%(exc_info)s\n'
))
error_logger.addHandler(error_handler)

# ============================================================================
# DATA MODELS AND STRUCTURES
# ============================================================================

@dataclass
class UserProfile:
    """User profile data structure"""
    user_id: int
    username: str = ""
    first_name: str = ""
    last_name: str = ""
    phone: str = ""
    role: str = "free"  # free, premium, admin, owner
    joined_date: str = ""
    last_active: str = ""
    total_forwards: int = 0
    total_rules: int = 0
    is_banned: bool = False
    ban_reason: str = ""
    premium_until: str = ""
    settings: Dict = None
    
    def __post_init__(self):
        if self.settings is None:
            self.settings = {
                "notifications": True,
                "auto_delete": False,
                "language": "en",
                "timezone": "UTC",
                "privacy_mode": "default" # default, strict
            }

@dataclass
class ForwardRule:
    """Forwarding rule data structure"""
    rule_id: str
    user_id: int
    source_chat: int
    target_chats: List[int]
    mode: str  # with_tags, without_tags, bypass
    keywords: List[str] = None
    exclude_keywords: List[str] = None
    replace_text: Dict[str, str] = None
    is_active: bool = True
    created_date: str = ""
    last_triggered: str = ""
    trigger_count: int = 0
    schedule: Dict = None # {"enabled": bool, "start_time": "HH:MM", "end_time": "HH:MM", "days": [0-6]}
    filters: Dict = None # {"media_only": bool, "text_only": bool, "min_length": int, "max_length": int, "allowed_types": List[str]}
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.exclude_keywords is None:
            self.exclude_keywords = []
        if self.replace_text is None:
            self.replace_text = {}
        if self.schedule is None:
            self.schedule = {"enabled": False, "start_time": "00:00", "end_time": "23:59", "days": list(range(7))}
        if self.filters is None:
            self.filters = {
                "media_only": False,
                "text_only": False,
                "min_length": 0,
                "max_length": 0,
                "allowed_types": []
            }

@dataclass
class Statistics:
    """Statistics data structure"""
    total_forwards: int = 0
    successful_forwards: int = 0
    failed_forwards: int = 0
    total_rules: int = 0
    active_rules: int = 0
    total_users: int = 0
    premium_users: int = 0
    admin_users: int = 0
    daily_stats: Dict = None # {"YYYY-MM-DD": {"forwards": X, "users": Y}}
    monthly_stats: Dict = None # {"YYYY-MM": {"forwards": X, "users": Y}}
    
    def __post_init__(self):
        if self.daily_stats is None:
            self.daily_stats = {}
        if self.monthly_stats is None:
            self.monthly_stats = {}

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Advanced database manager with SQLite backend"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    phone TEXT,
                    role TEXT DEFAULT 'free',
                    joined_date TEXT,
                    last_active TEXT,
                    total_forwards INTEGER DEFAULT 0,
                    total_rules INTEGER DEFAULT 0,
                    is_banned BOOLEAN DEFAULT FALSE,
                    ban_reason TEXT,
                    premium_until TEXT,
                    settings TEXT
                )
            ''')
            
            # Rules table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rules (
                    rule_id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    source_chat INTEGER,
                    target_chats TEXT,
                    mode TEXT,
                    keywords TEXT,
                    exclude_keywords TEXT,
                    replace_text TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_date TEXT,
                    last_triggered TEXT,
                    trigger_count INTEGER DEFAULT 0,
                    schedule TEXT,
                    filters TEXT
                )
            ''')
            
            # Statistics table (stores daily aggregates)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS statistics (
                    date TEXT PRIMARY KEY,
                    total_forwards INTEGER DEFAULT 0,
                    successful_forwards INTEGER DEFAULT 0,
                    failed_forwards INTEGER DEFAULT 0,
                    total_rules_created INTEGER DEFAULT 0,
                    total_users_joined INTEGER DEFAULT 0
                )
            ''')
            
            # Logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    user_id INTEGER,
                    action TEXT,
                    details TEXT,
                    source_chat INTEGER,
                    target_chats TEXT,
                    status TEXT,
                    error_message TEXT
                )
            ''')
            
            # Settings table (for global bot settings)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS global_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            
            conn.commit()
    
    def get_user(self, user_id: int) -> Optional[UserProfile]:
        """Get user profile by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            
            if row:
                settings = json.loads(row[13]) if row[13] else {}
                return UserProfile(
                    user_id=row[0], username=row[1], first_name=row[2],
                    last_name=row[3], phone=row[4], role=row[5],
                    joined_date=row[6], last_active=row[7],
                    total_forwards=row[8], total_rules=row[9],
                    is_banned=bool(row[10]), ban_reason=row[11],
                    premium_until=row[12], settings=settings
                )
        return None
    
    def save_user(self, user: UserProfile):
        """Save user profile"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO users 
                (user_id, username, first_name, last_name, phone, role,
                 joined_date, last_active, total_forwards, total_rules,
                 is_banned, ban_reason, premium_until, settings)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user.user_id, user.username, user.first_name, user.last_name,
                user.phone, user.role, user.joined_date, user.last_active,
                user.total_forwards, user.total_rules, user.is_banned,
                user.ban_reason, user.premium_until, json.dumps(user.settings)
            ))
            conn.commit()

    def get_all_users(self) -> List[UserProfile]:
        """Get all user profiles"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users')
            users = []
            for row in cursor.fetchall():
                settings = json.loads(row[13]) if row[13] else {}
                users.append(UserProfile(
                    user_id=row[0], username=row[1], first_name=row[2],
                    last_name=row[3], phone=row[4], role=row[5],
                    joined_date=row[6], last_active=row[7],
                    total_forwards=row[8], total_rules=row[9],
                    is_banned=bool(row[10]), ban_reason=row[11],
                    premium_until=row[12], settings=settings
                ))
            return users
    
    def get_rules(self, user_id: int = None) -> List[ForwardRule]:
        """Get forwarding rules"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute('SELECT * FROM rules WHERE user_id = ?', (user_id,))
            else:
                cursor.execute('SELECT * FROM rules')
            
            rules = []
            for row in cursor.fetchall():
                rules.append(ForwardRule(
                    rule_id=row[0], user_id=row[1], source_chat=row[2],
                    target_chats=json.loads(row[3]), mode=row[4],
                    keywords=json.loads(row[5]) if row[5] else [],
                    exclude_keywords=json.loads(row[6]) if row[6] else [],
                    replace_text=json.loads(row[7]) if row[7] else {},
                    is_active=bool(row[8]), created_date=row[9],
                    last_triggered=row[10], trigger_count=row[11],
                    schedule=json.loads(row[12]) if row[12] else {},
                    filters=json.loads(row[13]) if row[13] else {}
                ))
            return rules
    
    def get_rule(self, rule_id: str) -> Optional[ForwardRule]:
        """Get a single forwarding rule by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM rules WHERE rule_id = ?', (rule_id,))
            row = cursor.fetchone()
            if row:
                return ForwardRule(
                    rule_id=row[0], user_id=row[1], source_chat=row[2],
                    target_chats=json.loads(row[3]), mode=row[4],
                    keywords=json.loads(row[5]) if row[5] else [],
                    exclude_keywords=json.loads(row[6]) if row[6] else [],
                    replace_text=json.loads(row[7]) if row[7] else {},
                    is_active=bool(row[8]), created_date=row[9],
                    last_triggered=row[10], trigger_count=row[11],
                    schedule=json.loads(row[12]) if row[12] else {},
                    filters=json.loads(row[13]) if row[13] else {}
                )
        return None

    def save_rule(self, rule: ForwardRule):
        """Save forwarding rule"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO rules
                (rule_id, user_id, source_chat, target_chats, mode,
                 keywords, exclude_keywords, replace_text, is_active,
                 created_date, last_triggered, trigger_count, schedule, filters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.rule_id, rule.user_id, rule.source_chat,
                json.dumps(rule.target_chats), rule.mode,
                json.dumps(rule.keywords), json.dumps(rule.exclude_keywords),
                json.dumps(rule.replace_text), rule.is_active,
                rule.created_date, rule.last_triggered, rule.trigger_count,
                json.dumps(rule.schedule), json.dumps(rule.filters)
            ))
            conn.commit()
    
    def delete_rule(self, rule_id: str):
        """Delete forwarding rule"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM rules WHERE rule_id = ?', (rule_id,))
            conn.commit()
    
    def log_action(self, user_id: int, action: str, details: str = "",
                   source_chat: int = None, target_chats: List[int] = None,
                   status: str = "success", error_message: str = ""):
        """Log user action"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO logs
                (timestamp, user_id, action, details, source_chat,
                 target_chats, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(), user_id, action, details,
                source_chat, json.dumps(target_chats) if target_chats else None,
                status, error_message
            ))
            conn.commit()
    
    def get_logs(self, user_id: int = None, limit: int = 100) -> List[Dict]:
        """Get logs, optionally filtered by user_id"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute('SELECT * FROM logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?', (user_id, limit))
            else:
                cursor.execute('SELECT * FROM logs ORDER BY timestamp DESC LIMIT ?', (limit,))
            
            logs = []
            for row in cursor.fetchall():
                logs.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "user_id": row[2],
                    "action": row[3],
                    "details": row[4],
                    "source_chat": row[5],
                    "target_chats": json.loads(row[6]) if row[6] else [],
                    "status": row[7],
                    "error_message": row[8]
                })
            return logs

    def get_global_setting(self, key: str, default: Any = None) -> Any:
        """Get a global bot setting"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM global_settings WHERE key = ?', (key,))
            row = cursor.fetchone()
            if row:
                try:
                    return json.loads(row[0]) # Settings are stored as JSON strings
                except json.JSONDecodeError:
                    return row[0] # Return as string if not valid JSON
            return default

    def set_global_setting(self, key: str, value: Any):
        """Set a global bot setting"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO global_settings (key, value) VALUES (?, ?)
            ''', (key, json.dumps(value))) # Store all settings as JSON strings
            conn.commit()

    def get_all_global_settings(self) -> Dict[str, Any]:
        """Get all global bot settings"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT key, value FROM global_settings')
            settings = {}
            for row in cursor.fetchall():
                try:
                    settings[row[0]] = json.loads(row[1])
                except json.JSONDecodeError:
                    settings[row[0]] = row[1]
            return settings

    def get_statistics(self) -> Statistics:
        """Get aggregated bot statistics from daily records and current user/rule counts"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Aggregate from daily statistics table
            cursor.execute('''
                SELECT 
                    SUM(total_forwards), 
                    SUM(successful_forwards), 
                    SUM(failed_forwards)
                FROM statistics
            ''')
            agg_stats = cursor.fetchone()
            
            total_forwards = agg_stats[0] if agg_stats[0] is not None else 0
            successful_forwards = agg_stats[1] if agg_stats[1] is not None else 0
            failed_forwards = agg_stats[2] if agg_stats[2] is not None else 0

            # Get current counts from users and rules tables
            cursor.execute('SELECT COUNT(*) FROM users')
            total_users = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'premium'")
            premium_users = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
            admin_users = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM rules')
            total_rules = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM rules WHERE is_active = TRUE')
            active_rules = cursor.fetchone()[0]

            # Get daily stats for charting
            daily_stats_data = {}
            cursor.execute('SELECT date, total_forwards, successful_forwards, failed_forwards FROM statistics ORDER BY date DESC LIMIT 30')
            for row in cursor.fetchall():
                daily_stats_data[row[0]] = {
                    "forwards": row[1],
                    "successful": row[2],
                    "failed": row[3]
                }
            
            return Statistics(
                total_forwards=total_forwards,
                successful_forwards=successful_forwards,
                failed_forwards=failed_forwards,
                total_rules=total_rules,
                active_rules=active_rules,
                total_users=total_users,
                premium_users=premium_users,
                admin_users=admin_users,
                daily_stats=daily_stats_data
            )
    
    def update_daily_statistics(self, date: str, forwards_data: Dict[str, int], users_data: Dict[str, int]):
        """Update daily statistics for a given date."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO statistics
                (date, total_forwards, successful_forwards, failed_forwards, total_rules_created, total_users_joined)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                date,
                forwards_data.get('total', 0),
                forwards_data.get('successful', 0),
                forwards_data.get('failed', 0),
                users_data.get('rules_created', 0),
                users_data.get('users_joined', 0)
            ))
            conn.commit()

# Initialize database
db = DatabaseManager(DATABASE_URL)

# Initialize global settings from DB, with defaults
GLOBAL_SETTINGS = db.get_all_global_settings()
if not GLOBAL_SETTINGS: # Populate with initial defaults if DB is empty
    GLOBAL_SETTINGS = {
        "auto_forward": True,
        "phone_required": False,
        "rate_limit": RATE_LIMIT,
        "max_rules_per_user": 50,
        "premium_features": True,
        "webhook_enabled": bool(WEBHOOK_URL),
        "maintenance_mode": False,
        "debug_mode": False,
        "backup_enabled": True,
        "analytics_enabled": True,
        "notification_enabled": True,
        "broadcast_template": "📢 <b>Broadcast Message</b>\n\n{message}\n\n<i>Sent by bot admin</i>", # Changed to HTML
        "owner_console_enabled": True # For console access
    }
    for key, value in GLOBAL_SETTINGS.items():
        db.set_global_setting(key, value)

# ============================================================================
# UTILITY FUNCTIONS AND DECORATORS
# ============================================================================

def get_current_time() -> str:
    """Get current timestamp as ISO string"""
    return datetime.now().isoformat()

def generate_rule_id() -> str:
    """Generate unique rule ID"""
    return str(uuid.uuid4())

def validate_chat_id(chat_id: str) -> int:
    """Validate and convert chat ID"""
    if not chat_id.lstrip('-').isdigit():
        raise ValueError("Invalid chat ID format")
    return int(chat_id)

def format_duration(seconds: int) -> str:
    """Format duration in human readable format"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"

def _are_reply_markups_equal(m1, m2):
    """Helper to compare InlineKeyboardMarkup objects."""
    if m1 is None and m2 is None:
        return True
    if (m1 is None) != (m2 is None):
        return False
    try:
        # Convert to dict for comparison, handling potential differences in object identity
        return m1.to_dict() == m2.to_dict()
    except Exception:
        return False

async def safe_edit_message(query: CallbackQuery, text: str, reply_markup: Optional[InlineKeyboardMarkup] = None, parse_mode: ParseMode = ParseMode.HTML):
    """
    Safely edits a message, handling 'Message is not modified' errors.
    Ensures the message has editable text content.
    """
    current_message = query.message

    # Check if the message has text content that can be edited
    # For media messages, current_message.text might be None, but current_message.caption might exist.
    # We prioritize caption if text is None.
    current_text_content = current_message.text or current_message.caption

    # If the message has no text or caption, it cannot be edited with edit_message_text
    if current_text_content is None:
        try:
            await query.answer("⚠️ Cannot edit this type of message (no text/caption to edit).", show_alert=True)
        except Exception as e:
            logger.warning(f"Failed to answer query for uneditable message: {e}")
        return

    same_text = current_text_content == text
    same_markup = _are_reply_markups_equal(current_message.reply_markup, reply_markup)

    if same_text and same_markup:
        try:
            await query.answer("⚠️ Nothing changed.", show_alert=False)
        except Exception as e:
            logger.warning(f"Failed to answer query for no change: {e}")
        return

    try:
        await query.edit_message_text(text=text, reply_markup=reply_markup, parse_mode=parse_mode)
    except BadRequest as e:
        if "Message is not modified" in str(e):
            try:
                await query.answer("⚠️ Nothing changed.", show_alert=False)
            except Exception as e_inner:
                logger.warning(f"Failed to answer query for 'Message is not modified': {e_inner}")
        elif "There is no text in the message to edit" in str(e):
            try:
                await query.answer("⚠️ Cannot edit: the original message has no editable text.", show_alert=True)
            except Exception as e_inner:
                logger.warning(f"Failed to answer query for 'no text to edit': {e_inner}")
        elif "message to edit not found" in str(e):
            try:
                await query.answer("⚠️ Message to edit not found.", show_alert=True)
            except Exception as e_inner:
                logger.warning(f"Failed to answer query for 'message not found': {e_inner}")
        elif "Can't parse entities" in str(e):
            logger.error(f"HTML parsing error in safe_edit_message: {e}. Text: {text}")
            try:
                await query.answer("❌ Formatting error in message. Please contact support.", show_alert=True)
            except Exception as e_inner:
                logger.warning(f"Failed to answer query for 'parse entities error': {e_inner}")
            # Optionally, try sending without parse_mode or with a different one for debugging
            # await query.edit_message_text(text=text, reply_markup=reply_markup)
        else:
            logger.error(f"Unhandled BadRequest in safe_edit_message: {e}")
            raise # Re-raise other BadRequest errors

def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

def get_string(lang_code: str, key: str) -> str:
    """Multi-language string retrieval (simplified)"""
    # In a real app, this would load from JSON files
    strings = {
        "en": {
            "admin_privileges_required": "❌ Admin privileges required.",
            "owner_privileges_required": "❌ Owner privileges required.",
            "premium_subscription_required": "❌ Premium subscription required.",
            "rate_limit_exceeded": "⚠️ Rate limit exceeded. Please wait {time_window} seconds.",
            "phone_verification_required": "❌ Phone verification required. Use /connect_phone first.",
            "invalid_mode": "❌ Invalid mode. Use: <code>with_tags</code>, <code>without_tags</code>, or <code>bypass</code>",
            "invalid_chat_id_format": "❌ Invalid chat ID format: {error}",
            "forward_command_usage": "📝 <b>Forward Command Usage:</b>\n\n<code>/forward &lt;mode&gt; &lt;source_chat&gt; &lt;target_chat1&gt; [target_chat2] ...</code>\n\n<b>Modes:</b>\n• <code>with_tags</code> - Keep original sender info\n• <code>without_tags</code> - Clean forwarding\n• <code>bypass</code> - Bypass restrictions\n\n<b>Example:</b>\n<code>/forward without_tags -1001234567890 -1009876543210</code>",
            "autoforward_command_usage": "📝 <b>Auto-Forward Command Usage:</b>\n\n<code>/autoforward &lt;mode&gt; &lt;source_chat&gt; &lt;target_chat1&gt; [target_chat2] ...</code>\n\n<b>This creates a permanent forwarding rule.</b>\n\n<b>Example:</b>\n<code>/autoforward without_tags -1001234567890 -1009876543210</code>",
            "max_rules_limit_reached": "❌ Maximum rules limit reached ({max_rules}). Delete some rules first.",
            "rule_created_success": "✅ <b>Auto-Forward Rule Created!</b>",
            "no_rules_found": "📝 <b>No Forwarding Rules</b>\n\nYou haven't created any forwarding rules yet.\nUse /autoforward to create your first rule!\n\n<b>Example:</b>\n<code>/autoforward without_tags -1001234567890 -1009876543210</code>",
            "your_rules": "📝 <b>Your Forwarding Rules ({count})</b>",
            "no_stats_available": "❌ No statistics available.",
            "your_stats": "📊 <b>Your Statistics</b>",
            "users_management": "👥 <b>Users Management</b>",
            "promote_command_usage": "📝 <b>Promote User Usage:</b>\n\n<code>/promote &lt;user_id&gt; &lt;role&gt;</code>\n\n<b>Available roles:</b>\n• <code>premium</code>\n• <code>admin</code> (owner only)\n\n<b>Example:</b>\n<code>/promote 123456789 premium</code>",
            "invalid_user_id_format": "❌ Invalid user ID format.",
            "invalid_role": "❌ Invalid role. Use 'premium' or 'admin'.",
            "owner_only_admin_promote": "❌ Only owner can promote to admin.",
            "user_promoted_success": "✅ <b>User Promoted!</b>",
            "failed_to_promote_user": "❌ Failed to promote user.",
            "system_stats": "🖥️ <b>System Statistics</b>",
            "bot_settings": "⚙️ <b>Bot Settings</b>",
            "maintenance_mode_info": "🔧 <b>Maintenance Mode</b>",
            "system_backup_info": "💾 <b>System Backup</b>",
            "debug_info": "🐛 <b>Debug Information</b>",
            "bot_under_maintenance": "🚧 The bot is currently under maintenance. Please try again later.",
            "phone_already_verified": "✅ Your phone is already verified!",
            "phone_verification_text": "📱 <b>Phone Verification</b>\n\nTo verify your phone number:\n1. Click the button below\n2. Allow Telegram to share your number\n3. You'll receive a verification code\n\nThis is required for premium features!",
            "share_own_phone": "❌ Please share your own phone number.",
            "phone_verified_success": "✅ <b>Phone Verified Successfully!</b>",
            "forward_processing": "⏳ <b>Processing forward...</b>",
            "forward_complete": "✅ <b>Forward Complete!</b>",
            "forward_failed": "❌ <b>Forward Failed</b>",
            "rule_settings_not_found": "❌ Rule not found.",
            "rule_settings_menu": "⚙️ <b>Rule Settings for Rule {rule_id_short}</b>",
            "rule_toggle_success": "✅ Rule {rule_id_short} is now {status}.",
            "rule_deleted_success": "✅ Rule {rule_id_short} deleted.",
            "confirm_delete_rule": "⚠️ Are you sure you want to delete rule {rule_id_short}?",
            "rule_edit_keywords": "📝 <b>Edit Keywords for Rule {rule_id_short}</b>\n\n<b>Current Keywords:</b> {keywords}\n\nSend new keywords separated by commas. Use <code>/cancel</code> to abort.",
            "rule_edit_exclude_keywords": "📝 <b>Edit Exclude Keywords for Rule {rule_id_short}</b>\n\n<b>Current Exclude Keywords:</b> {exclude_keywords}\n\nSend new exclude keywords separated by commas. Use <code>/cancel</code> to abort.",
            "rule_edit_replace_text": "📝 <b>Edit Replace Text for Rule {rule_id_short}</b>\n\n<b>Current Replacements:</b> {replace_text}\n\nSend replacements in <code>old_text=new_text</code> format, one per line. Use <code>/cancel</code> to abort.",
            "rule_edit_filters": "⚙️ <b>Edit Filters for Rule {rule_id_short}</b>",
            "rule_edit_schedule": "⏰ <b>Edit Schedule for Rule {rule_id_short}</b>",
            "rule_edit_targets": "🎯 <b>Edit Target Chats for Rule {rule_id_short}</b>\n\n<b>Current Targets:</b> {targets}\n\nSend new target chat IDs separated by commas. Use <code>/cancel</code> to abort.",
            "rule_edit_mode": "🔄 <b>Change Mode for Rule {rule_id_short}</b>\n\n<b>Current Mode:</b> {mode}\n\nSelect a new mode:",
            "rule_edit_success": "✅ Rule {rule_id_short} updated successfully!",
            "broadcast_command_usage": "📢 <b>Broadcast Message Usage:</b>\n\n<code>/broadcast &lt;message&gt;</code> - Send to all users\n<code>/broadcast_role &lt;role&gt; &lt;message&gt;</code> - Send to specific role (e.g., <code>premium</code>, <code>admin</code>)\n\n<b>Example:</b>\n<code>/broadcast Hello everyone!</code>",
            "broadcast_sent": "✅ Broadcast sent to {count} users.",
            "broadcast_failed": "❌ Failed to send broadcast to some users. Check logs.",
            "user_not_found": "❌ User not found.",
            "user_info": "👤 <b>User Information for {user_name}</b>",
            "user_banned_success": "✅ User {user_id} banned.",
            "user_unbanned_success": "✅ User {user_id} unbanned.",
            "ban_command_usage": "🚫 <b>Ban User Usage:</b>\n\n<code>/ban &lt;user_id&gt; [reason]</code>\n\n<b>Example:</b>\n<code>/ban 123456789 Spamming</code>",
            "unban_command_usage": "✅ <b>Unban User Usage:</b>\n\n<code>/unban &lt;user_id&gt;</code>\n\n<b>Example:</b>\n<code>/unban 123456789</code>",
            "settings_updated": "✅ Setting '{key}' updated to '{value}'.",
            "invalid_setting_value": "❌ Invalid value for setting '{key}'.",
            "backup_created": "✅ Backup created: {backup_name} and {db_backup_name}",
            "backup_failed": "❌ Backup failed: {error}",
            "restore_command_usage": "🔄 <b>Restore Backup Usage:</b>\n\n<code>/restore &lt;backup_file_name&gt;</code>\n\n<b>Example:</b>\n<code>/restore backup_20240120_120000.json</code>\n\n<b>Warning: This will overwrite current data!</b>",
            "restore_success": "✅ Database restored from {backup_file_name}. Bot will restart.",
            "restore_failed": "❌ Restore failed: {error}",
            "no_backups_found": "No backups found.",
            "available_backups": "💾 <b>Available Backups:</b>",
            "clear_logs_success": "✅ Logs cleared up to 30 days ago.",
            "clear_logs_failed": "❌ Failed to clear logs: {error}",
            "owner_console_disabled": "❌ Owner console is disabled in settings.",
            "cancel_operation": "Operation cancelled.",
            "invalid_input": "❌ Invalid input. Please try again or /cancel.",
            "rule_edit_media_types": "🖼️ <b>Edit Allowed Media Types for Rule {rule_id_short}</b>\n\n<b>Current Types:</b> {types}\n\nSend new types separated by commas (e.g., <code>photo,video,text</code>). Use <code>/cancel</code> to abort.\n\n<b>Available types:</b> <code>text, photo, video, document, audio, voice, sticker, animation, poll, contact, location, venue, video_note</code>",
            "rule_edit_length_filters": "📏 <b>Edit Message Length Filters for Rule {rule_id_short}</b>\n\n<b>Current Min Length:</b> {min_len}\n<b>Current Max Length:</b> {max_len}\n\nSend min and max length separated by a comma (e.g., <code>10,500</code>). Set to <code>0,0</code> to disable. Use <code>/cancel</code> to abort.",
            "rule_edit_schedule_time": "⏰ <b>Edit Schedule Time for Rule {rule_id_short}</b>\n\n<b>Current Time:</b> {start_time} - {end_time}\n\nSend start and end time in HH:MM format separated by a hyphen (e.g., <code>09:00-17:00</code>). Use <code>/cancel</code> to abort.",
            "rule_edit_schedule_days": "🗓️ <b>Edit Schedule Days for Rule {rule_id_short}</b>\n\n<b>Current Days:</b> {days}\n\nSend days of the week separated by commas (e.g., <code>Mon,Wed,Fri</code>). Use <code>/cancel</code> to abort.\n\n<b>Days:</b> <code>Mon, Tue, Wed, Thu, Fri, Sat, Sun</code>",
            "rule_edit_schedule_toggle": "Toggle schedule for rule {rule_id_short}.",
            "my_rules": "My Rules", # Added for inline button
            "rule_settings_menu": "Rule Settings for Rule {rule_id_short}", # Added for inline button
            "Share Phone Number": "Share Phone Number" # Added for inline button
        },
        # Add other languages here
        "es": {
            "admin_privileges_required": "❌ Se requieren privilegios de administrador.",
            # ... (add all translated strings) ...
        }
    }
    # Escape all strings for HTML to ensure proper parsing.
    # This is a broad stroke; for more fine-grained control, you'd escape parts of strings.
    # For now, it ensures the bot doesn't crash due to unescaped characters.
    text = strings.get(lang_code, strings["en"]).get(key, key)
    return escape_html(text)


# Conversation states for rule creation/editing
(
    CREATE_RULE_MODE, CREATE_RULE_SOURCE, CREATE_RULE_TARGETS,
    EDIT_RULE_SELECT, EDIT_RULE_KEYWORDS, EDIT_RULE_EXCLUDE_KEYWORDS,
    EDIT_RULE_REPLACE_TEXT, EDIT_RULE_FILTERS, EDIT_RULE_SCHEDULE,
    EDIT_RULE_TARGETS, EDIT_RULE_MODE, EDIT_RULE_MEDIA_TYPES,
    EDIT_RULE_LENGTH_FILTERS, EDIT_RULE_SCHEDULE_TIME, EDIT_RULE_SCHEDULE_DAYS,
    # Renamed to avoid conflict with global vars if they were used elsewhere
    BROADCAST_MESSAGE_STATE, BROADCAST_ROLE_MESSAGE_STATE, 
    SETTING_CHANGE_VALUE,
    BAN_USER_REASON
) = range(19)


# ============================================================================
# ROLE MANAGEMENT SYSTEM
# ============================================================================

class RoleManager:
    """Advanced role management system"""
    
    ROLES = {
        'free': 0,
        'premium': 1,
        'admin': 2,
        'owner': 3
    }
    
    @staticmethod
    def get_role(user_id: int) -> str:
        """Get user role"""
        if user_id == OWNER_ID:
            return 'owner'
        
        user_profile = db.get_user(user_id)
        if user_profile:
            return user_profile.role
        
        return 'free'
    
    @staticmethod
    def has_permission(user_id: int, required_role: str) -> bool:
        """Check if user has required permission level"""
        user_role = RoleManager.get_role(user_id)
        return RoleManager.ROLES[user_role] >= RoleManager.ROLES[required_role]
    
    @staticmethod
    def is_owner(user_id: int) -> bool:
        return user_id == OWNER_ID
    
    @staticmethod
    def is_admin(user_id: int) -> bool:
        return RoleManager.has_permission(user_id, 'admin')
    
    @staticmethod
    def is_premium(user_id: int) -> bool:
        return RoleManager.has_permission(user_id, 'premium')
    
    @staticmethod
    def promote_user(user_id: int, role: str) -> bool:
        """Promote user to specified role"""
        if role not in RoleManager.ROLES or RoleManager.ROLES[role] > RoleManager.ROLES['admin']:
            return False # Cannot promote to owner via this method
        
        user_profile = db.get_user(user_id)
        if not user_profile:
            # Create new user profile if it doesn't exist
            user_profile = UserProfile(
                user_id=user_id,
                role=role,
                joined_date=get_current_time(),
                last_active=get_current_time()
            )
        else:
            user_profile.role = role
        
        db.save_user(user_profile)
        return True
    
    @staticmethod
    def demote_user(user_id: int, role: str = 'free') -> bool:
        """Demote user to specified role"""
        if role not in RoleManager.ROLES:
            return False
        
        user_profile = db.get_user(user_id)
        if user_profile:
            user_profile.role = role
            db.save_user(user_profile)
            return True
        return False

# Decorators for role-based access
def admin_required(func):
    """Decorator to require admin privileges"""
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
        
        if not RoleManager.is_admin(user_id):
            if update.message:
                await update.message.reply_text(get_string(user_lang, "admin_privileges_required"))
            elif update.callback_query:
                await update.callback_query.answer(get_string(user_lang, "admin_privileges_required"), show_alert=True)
            return
        
        return await func(update, context)
    return wrapper

def owner_required(func):
    """Decorator to require owner privileges"""
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
        
        if not RoleManager.is_owner(user_id):
            if update.message:
                await update.message.reply_text(get_string(user_lang, "owner_privileges_required"))
            elif update.callback_query:
                await update.callback_query.answer(get_string(user_lang, "owner_privileges_required"), show_alert=True)
            return
        
        return await func(update, context)
    return wrapper

def premium_required(func):
    """Decorator to require premium privileges"""
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
        
        if not RoleManager.is_premium(user_id):
            if update.message:
                await update.message.reply_text(get_string(user_lang, "premium_subscription_required"))
            elif update.callback_query:
                await update.callback_query.answer(get_string(user_lang, "premium_subscription_required"), show_alert=True)
            return
        
        return await func(update, context)
    return wrapper

def rate_limit(max_calls: int = 5, time_window: int = 60):
    """Rate limiting decorator"""
    call_history = defaultdict(list)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
            user_id = update.effective_user.id
            user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
            now = time.time()
            
            # Clean old calls
            call_history[user_id] = [
                call_time for call_time in call_history[user_id]
                if now - call_time < time_window
            ]
            
            # Check rate limit
            if len(call_history[user_id]) >= max_calls:
                if update.message:
                    await update.message.reply_text(
                        get_string(user_lang, "rate_limit_exceeded").format(time_window=time_window)
                    )
                elif update.callback_query:
                    await update.callback_query.answer(
                        get_string(user_lang, "rate_limit_exceeded").format(time_window=time_window), show_alert=True
                    )
                return
            
            # Record this call
            call_history[user_id].append(now)
            
            return await func(update, context)
        return wrapper
    return decorator

# ============================================================================
# FORWARDING ENGINE
# ============================================================================

class ForwardingEngine:
    """Advanced message forwarding engine"""
    
    def __init__(self):
        self.active_rules = {}
        self.stats_in_memory = { # Temporary in-memory stats for current session
            'total_forwards': 0,
            'successful_forwards': 0,
            'failed_forwards': 0,
            'rate_limited': 0
        }
        self.load_active_rules()
    
    def load_active_rules(self):
        """Load active forwarding rules"""
        self.active_rules = {}
        try:
            rules = db.get_rules()
            
            for rule in rules:
                if rule.is_active:
                    if rule.source_chat not in self.active_rules:
                        self.active_rules[rule.source_chat] = []
                    self.active_rules[rule.source_chat].append(rule)
            
            logger.info(f"Loaded {len(rules)} total rules, {sum(len(v) for v in self.active_rules.values())} active rules.")
        except Exception as e:
            logger.error(f"Error loading rules: {e}")
            self.active_rules = {} # Initialize empty to prevent crashes
    
    def reload_rules(self):
        """Reload forwarding rules"""
        self.load_active_rules()
    
    async def process_message(self, update, context):
      """Process incoming message for forwarding"""
      # Handle both regular messages and channel posts
      message = update.effective_message or update.channel_post
      if not message:
        return
    
      chat_id = message.chat_id
    
      if chat_id not in self.active_rules:
        return
    
      for rule in self.active_rules[chat_id]:
        # Check if rule is active based on schedule
        if rule.schedule["enabled"]:
            now = datetime.now()
            current_time = now.time()
            current_day = now.weekday()

            start_time_str = rule.schedule.get("start_time", "00:00")
            end_time_str = rule.schedule.get("end_time", "23:59")
            scheduled_days = rule.schedule.get("days", list(range(7)))

            try:
                start_time = datetime.strptime(start_time_str, "%H:%M").time()
                end_time = datetime.strptime(end_time_str, "%H:%M").time()
            except ValueError:
                logger.warning(f"Invalid schedule time format for rule {rule.rule_id}")
                rule.schedule["enabled"] = False
                db.save_rule(rule)
                continue

            if current_day not in scheduled_days:
                continue
            
            if start_time <= end_time:
                if not (start_time <= current_time <= end_time):
                    continue
            else:
                if not (current_time >= start_time or current_time <= end_time):
                    continue
        
        try:
            if await self._should_forward(message, rule):
                await self._forward_message(message, rule, context.bot)
                
                # Update rule statistics
                rule.trigger_count += 1
                rule.last_triggered = get_current_time()
                db.save_rule(rule)
                
        except Exception as e:
            logger.error(f"Error processing rule {rule.rule_id}: {e}")
            self.stats_in_memory['failed_forwards'] += 1
            db.log_action(
                user_id=rule.user_id,
                action='auto_forward',
                details=f"Rule {rule.rule_id}",
                status='error',
                source_chat=message.chat_id,
                target_chats=rule.target_chats,
                error_message=str(e)
            )

    
    async def _should_forward(self, message, rule) -> bool:
        """Check if message should be forwarded based on rule criteria"""
        
        text_content = message.text or message.caption or ""
        
        # Check include keywords
        if rule.keywords:
            keyword_found = any(
                keyword.strip().lower() in text_content.lower()
                for keyword in rule.keywords if keyword.strip()
            )
            if not keyword_found:
                return False
        
        # Check exclude keywords
        if rule.exclude_keywords:
            exclude_found = any(
                keyword.strip().lower() in text_content.lower()
                for keyword in rule.exclude_keywords if keyword.strip()
            )
            if exclude_found:
                return False
        
        # Apply filters
        if rule.filters:
            filters = rule.filters
            
            if filters.get("media_only") and not (message.photo or message.video or message.document or message.audio or message.voice or message.sticker or message.animation or message.video_note):
                return False
            if filters.get("text_only") and not (message.text or message.caption):
                return False
            
            if filters.get("min_length") and len(text_content) < filters["min_length"]:
                return False
            if filters.get("max_length") and filters["max_length"] > 0 and len(text_content) > filters["max_length"]:
                return False
            
            allowed_types = filters.get("allowed_types")
            if allowed_types:
                message_type = None
                if message.text or message.caption: message_type = "text"
                elif message.photo: message_type = "photo"
                elif message.video: message_type = "video"
                elif message.document: message_type = "document"
                elif message.audio: message_type = "audio"
                elif message.voice: message_type = "voice"
                elif message.sticker: message_type = "sticker"
                elif message.animation: message_type = "animation"
                elif message.poll: message_type = "poll"
                elif message.contact: message_type = "contact"
                elif message.location: message_type = "location"
                elif message.venue: message_type = "venue"
                elif message.video_note: message_type = "video_note"
                
                if message_type and message_type not in allowed_types:
                    return False

        return True
    
    async def _forward_message(self, message: Message, rule: ForwardRule, bot: Bot):
        """Forward message to target chats"""
        self.stats_in_memory['total_forwards'] += 1
        
        target_chats = rule.target_chats
        
        for target_chat in target_chats:
            try:
                processed_text = message.text or message.caption
                if processed_text and rule.replace_text:
                    for old, new in rule.replace_text.items():
                        processed_text = processed_text.replace(old, new)

                if rule.mode == 'with_tags':
                    await bot.forward_message(
                        chat_id=target_chat,
                        from_chat_id=message.chat_id,
                        message_id=message.message_id
                    )
                else:  # without_tags or bypass
                    # Use copy_message for clean forwarding, allowing caption modification
                    # Fallback to send_message for text-only, or specific media types
                    # This is a more robust way to handle various message types
                    if message.text:
                        await bot.send_message(
                            chat_id=target_chat,
                            text=processed_text,
                            parse_mode=ParseMode.HTML, # Use HTML
                            entities=message.entities if message.entities and processed_text == (message.text or message.caption) else None
                        )
                    elif message.photo:
                        await bot.send_photo(
                            chat_id=target_chat,
                            photo=message.photo[-1].file_id,
                            caption=processed_text,
                            parse_mode=ParseMode.HTML, # Use HTML
                            caption_entities=message.caption_entities if message.caption_entities and processed_text == (message.text or message.caption) else None
                        )
                    elif message.video:
                        await bot.send_video(
                            chat_id=target_chat,
                            video=message.video.file_id,
                            caption=processed_text,
                            parse_mode=ParseMode.HTML, # Use HTML
                            caption_entities=message.caption_entities if message.caption_entities and processed_text == (message.text or message.caption) else None
                        )
                    elif message.document:
                        await bot.send_document(
                            chat_id=target_chat,
                            document=message.document.file_id,
                            caption=processed_text,
                            parse_mode=ParseMode.HTML, # Use HTML
                            caption_entities=message.caption_entities if message.caption_entities and processed_text == (message.text or message.caption) else None
                        )
                    elif message.audio:
                        await bot.send_audio(
                            chat_id=target_chat,
                            audio=message.audio.file_id,
                            caption=processed_text,
                            parse_mode=ParseMode.HTML, # Use HTML
                            caption_entities=message.caption_entities if message.caption_entities and processed_text == (message.text or message.caption) else None
                        )
                    elif message.voice:
                        await bot.send_voice(
                            chat_id=target_chat,
                            voice=message.voice.file_id,
                            caption=processed_text,
                            parse_mode=ParseMode.HTML, # Use HTML
                            caption_entities=message.caption_entities if message.caption_entities and processed_text == (message.text or message.caption) else None
                        )
                    elif message.sticker:
                        await bot.send_sticker(
                            chat_id=target_chat,
                            sticker=message.sticker.file_id
                        )
                    elif message.animation:
                        await bot.send_animation(
                            chat_id=target_chat,
                            animation=message.animation.file_id,
                            caption=processed_text,
                            parse_mode=ParseMode.HTML, # Use HTML
                            caption_entities=message.caption_entities if message.caption_entities and processed_text == (message.text or message.caption) else None
                        )
                    elif message.poll:
                        await bot.send_poll(
                            chat_id=target_chat,
                            question=message.poll.question,
                            options=[o.text for o in message.poll.options],
                            is_anonymous=message.poll.is_anonymous,
                            type=message.poll.type,
                            allows_multiple_answers=message.poll.allows_multiple_answers,
                            correct_option_id=message.poll.correct_option_id,
                            explanation=message.poll.explanation,
                            explanation_parse_mode=ParseMode.HTML, # Use HTML
                            open_period=message.poll.open_period,
                            close_date=message.poll.close_date
                        )
                    elif message.contact:
                        await bot.send_contact(
                            chat_id=target_chat,
                            phone_number=message.contact.phone_number,
                            first_name=message.contact.first_name,
                            last_name=message.contact.last_name,
                            vcard=message.contact.vcard
                        )
                    elif message.location:
                        await bot.send_location(
                            chat_id=target_chat,
                            latitude=message.location.latitude,
                            longitude=message.location.longitude
                        )
                    elif message.venue:
                        await bot.send_venue(
                            chat_id=target_chat,
                            latitude=message.venue.location.latitude,
                            longitude=message.venue.location.longitude,
                            title=message.venue.title,
                            address=message.venue.address,
                            foursquare_id=message.venue.foursquare_id,
                            foursquare_type=message.venue.foursquare_type
                        )
                    elif message.video_note:
                        await bot.send_video_note(
                            chat_id=target_chat,
                            video_note=message.video_note.file_id
                        )
                    else:
                        # Fallback for unsupported message types, try copy_message
                        await bot.copy_message(
                            chat_id=target_chat,
                            from_chat_id=message.chat_id,
                            message_id=message.message_id
                        )
                
                self.stats_in_memory['successful_forwards'] += 1
                
                db.log_action(
                    user_id=rule.user_id,
                    action='auto_forward',
                    details=f"Rule {rule.rule_id}",
                    status='success',
                    source_chat=message.chat_id,
                    target_chats=[target_chat]
                )
                
            except Exception as e:
                self.stats_in_memory['failed_forwards'] += 1
                logger.error(f"Failed to forward to {target_chat}: {e}")
                
                db.log_action(
                    user_id=rule.user_id,
                    action='auto_forward',
                    details=f"Rule {rule.rule_id}",
                    status='error',
                    source_chat=message.chat_id,
                    target_chats=[target_chat],
                    error_message=str(e)
                )

    async def manual_forward_message(self, bot: Bot, message: Message, mode: str, target_chats: List[int], user_id: int):
        """
        Performs a manual one-time forward of a specific message.
        """
        results = {'successful': 0, 'failed': 0, 'errors': []}
        
        for target_chat in target_chats:
            try:
                if mode == 'with_tags':
                    await bot.forward_message(
                        chat_id=target_chat,
                        from_chat_id=message.chat_id,
                        message_id=message.message_id
                    )
                else: # without_tags or bypass
                    # Use copy_message for clean forwarding
                    await bot.copy_message(
                        chat_id=target_chat,
                        from_chat_id=message.chat_id,
                        message_id=message.message_id
                    )
                results['successful'] += 1
                db.log_action(user_id, 'manual_forward', f"Message {message.message_id} from {message.chat_id}", message.chat_id, [target_chat], 'success')
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"To {target_chat}: {e}")
                db.log_action(user_id, 'manual_forward', f"Message {message.message_id} from {message.chat_id}", message.chat_id, [target_chat], 'error', str(e))
        
        return results

# Initialize forwarding engine
forwarding_engine = ForwardingEngine()

# ============================================================================
# PHONE VERIFICATION SYSTEM
# ============================================================================

class PhoneVerification:
    """Phone verification system"""
    
    @staticmethod
    def is_required() -> bool:
        """Check if phone verification is required"""
        return GLOBAL_SETTINGS.get('phone_required', False)
    
    @staticmethod
    def has_phone(user_id: int) -> bool:
        """Check if user has verified phone"""
        user_profile = db.get_user(user_id)
        if user_profile and user_profile.phone:
            return True
        return False
    
    @staticmethod
    def save_phone(user_id: int, phone: str):
        """Save user phone number"""
        user_profile = db.get_user(user_id)
        if not user_profile:
            user_profile = UserProfile(
                user_id=user_id,
                phone=phone,
                joined_date=get_current_time(),
                last_active=get_current_time()
            )
        else:
            user_profile.phone = phone
        
        db.save_user(user_profile)
    
    @staticmethod
    def check_access(user_id: int) -> bool:
        """Check if user has access (considering phone requirement)"""
        if not PhoneVerification.is_required():
            return True
        
        if RoleManager.is_owner(user_id):
            return True
        
        return PhoneVerification.has_phone(user_id)

# ============================================================================
# CONVERSATION HANDLERS (for multi-step commands)
# ============================================================================

async def cancel_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    
    if update.message:
        await update.message.reply_text(
            get_string(user_lang, "cancel_operation"),
            reply_markup=ReplyKeyboardRemove()
        )
    elif update.callback_query:
        await update.callback_query.answer()
        await safe_edit_message(
            update.callback_query,
            get_string(user_lang, "cancel_operation"),
            reply_markup=None
        )
    
    context.user_data.clear() # Clear any stored data for the conversation
    return ConversationHandler.END

# --- Rule Creation Conversation ---
async def create_rule_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    
    # Use effective_message for consistency
    message_to_reply = update.effective_message

    if not PhoneVerification.check_access(user_id):
        await message_to_reply.reply_text(get_string(user_lang, "phone_verification_required"))
        return ConversationHandler.END

    user_rules = db.get_rules(user_id)
    max_rules = GLOBAL_SETTINGS.get('max_rules_per_user', 50)
    
    if len(user_rules) >= max_rules and not RoleManager.is_admin(user_id):
        await message_to_reply.reply_text(get_string(user_lang, "max_rules_limit_reached").format(max_rules=max_rules))
        return ConversationHandler.END

    keyboard = [
        [InlineKeyboardButton("with_tags", callback_data="mode_with_tags")],
        [InlineKeyboardButton("without_tags", callback_data="mode_without_tags")],
        [InlineKeyboardButton("bypass", callback_data="mode_bypass")],
        [InlineKeyboardButton(get_string(user_lang, "cancel_operation"), callback_data="cancel_conversation")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await message_to_reply.reply_text(
        get_string(user_lang, "autoforward_command_usage").split('\n\n')[0] + "\n\n" +
        "Please select a forwarding mode:",
        parse_mode=ParseMode.HTML, # Changed to HTML
        reply_markup=reply_markup
    )
    return CREATE_RULE_MODE

async def create_rule_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'

    if query.data == "cancel_conversation":
        return await cancel_conversation(update, context)

    mode = query.data.replace("mode_", "")
    context.user_data['new_rule_mode'] = mode

    await safe_edit_message(
        query,
        f"Selected mode: <code>{escape_html(mode)}</code>\n\n" # Changed to HTML
        "Now, please send the <b>source chat ID</b> (e.g., <code>-1001234567890</code>).", # Changed to HTML
        parse_mode=ParseMode.HTML # Changed to HTML
    )
    return CREATE_RULE_SOURCE

async def create_rule_source(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'

    try:
        source_chat = validate_chat_id(update.message.text.strip())
        context.user_data['new_rule_source'] = source_chat
    except ValueError as e:
        await update.message.reply_text(get_string(user_lang, "invalid_chat_id_format").format(error=escape_html(str(e)))) # Changed to HTML
        return CREATE_RULE_SOURCE # Stay in this state

    await update.message.reply_text(
        f"Source chat: <code>{escape_html(str(source_chat))}</code>\n\n" # Changed to HTML
        "Finally, send the <b>target chat IDs</b>, separated by commas (e.g., <code>-1009876543210, -100111222333</code>).", # Changed to HTML
        parse_mode=ParseMode.HTML # Changed to HTML
    )
    return CREATE_RULE_TARGETS

async def create_rule_targets(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'

    try:
        target_chats_str = update.message.text.strip()
        target_chats = [validate_chat_id(chat.strip()) for chat in target_chats_str.split(',') if chat.strip()]
        if not target_chats:
            raise ValueError("No valid target chats provided.")
        context.user_data['new_rule_targets'] = target_chats
    except ValueError as e:
        await update.message.reply_text(get_string(user_lang, "invalid_chat_id_format").format(error=escape_html(str(e)))) # Changed to HTML
        return CREATE_RULE_TARGETS # Stay in this state

    mode = context.user_data['new_rule_mode']
    source_chat = context.user_data['new_rule_source']
    target_chats = context.user_data['new_rule_targets']

    rule = ForwardRule(
        rule_id=generate_rule_id(),
        user_id=user_id,
        source_chat=source_chat,
        target_chats=target_chats,
        mode=mode,
        created_date=get_current_time()
    )
    
    db.save_rule(rule)
    forwarding_engine.reload_rules()
    
    user_profile = db.get_user(user_id)
    if user_profile:
        user_profile.total_rules += 1
        db.save_user(user_profile)
    
    success_text = get_string(user_lang, "rule_created_success") + f"""

🆔 <b>Rule ID:</b> <code>{escape_html(rule.rule_id)}</code>
📤 <b>Source:</b> <code>{escape_html(str(source_chat))}</code>
🎯 <b>Targets:</b> {', '.join(f'<code>{escape_html(str(t))}</code>' for t in target_chats)}
🔄 <b>Mode:</b> {escape_html(mode)}
📅 <b>Created:</b> {escape_html(datetime.fromisoformat(rule.created_date).strftime('%Y-%m-%d %H:%M'))}

🎉 Your rule is now active and will automatically forward new messages!

Use /rules to manage your forwarding rules.
"""
    
    keyboard = [
        [
            InlineKeyboardButton(get_string(user_lang, "my_rules"), callback_data="my_rules"),
            InlineKeyboardButton(get_string(user_lang, "rule_settings_menu").format(rule_id_short=escape_html(rule.rule_id[:8])), callback_data=f"rule_settings_{rule.rule_id}") # Changed to HTML
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        success_text,
        parse_mode=ParseMode.HTML, # Changed to HTML
        reply_markup=reply_markup
    )
    context.user_data.clear()
    return ConversationHandler.END

# --- Rule Editing Conversation ---

async def edit_rule_start(update: Update, context: ContextTypes.DEFAULT_TYPE, rule_id: str = None) -> int:
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'

    if not rule_id and context.args:
        rule_id = context.args[0]
    elif not rule_id and 'rule_id' in context.user_data:
        rule_id = context.user_data['rule_id']
    elif not rule_id:
        # Use effective_message for consistency
        message_to_reply = update.effective_message
        if message_to_reply:
            await message_to_reply.reply_text("Please provide a rule ID or select one from /rules.")
        return ConversationHandler.END

    rule = db.get_rule(rule_id)
    if not rule or rule.user_id != user_id:
        # Use effective_message for consistency
        message_to_reply = update.effective_message
        if message_to_reply:
            await message_to_reply.reply_text(get_string(user_lang, "rule_settings_not_found"))
        return ConversationHandler.END

    context.user_data['rule_id'] = rule_id
    context.user_data['current_rule'] = rule # Store the rule object

    keyboard = [
        [InlineKeyboardButton("🔄 Toggle Active", callback_data=f"edit_rule_toggle_active_{rule_id}")],
        [InlineKeyboardButton("📝 Keywords", callback_data=f"edit_rule_keywords_{rule_id}"),
         InlineKeyboardButton("🚫 Exclude Keywords", callback_data=f"edit_rule_exclude_keywords_{rule_id}")],
        [InlineKeyboardButton("✍️ Replace Text", callback_data=f"edit_rule_replace_text_{rule_id}"),
         InlineKeyboardButton("⚙️ Filters", callback_data=f"edit_rule_filters_{rule_id}")],
        [InlineKeyboardButton("⏰ Schedule", callback_data=f"edit_rule_schedule_{rule_id}"),
         InlineKeyboardButton("🎯 Target Chats", callback_data=f"edit_rule_targets_{rule_id}")],
        [InlineKeyboardButton("🔄 Mode", callback_data=f"edit_rule_mode_{rule_id}")],
        [InlineKeyboardButton("🗑️ Delete Rule", callback_data=f"edit_rule_delete_{rule_id}")],
        [InlineKeyboardButton("🔙 Back to Rules", callback_data="my_rules")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    status = "🟢 Active" if rule.is_active else "🔴 Inactive"
    text = get_string(user_lang, "rule_settings_menu").format(rule_id_short=escape_html(rule_id[:8])) + f"""

• <b>Status:</b> {escape_html(status)}
• <b>Source:</b> <code>{escape_html(str(rule.source_chat))}</code>
• <b>Targets:</b> {escape_html(str(len(rule.target_chats)))}
• <b>Mode:</b> {escape_html(rule.mode)}
• <b>Keywords:</b> {escape_html(', '.join(rule.keywords) if rule.keywords else 'None')}
• <b>Exclude Keywords:</b> {escape_html(', '.join(rule.exclude_keywords) if rule.exclude_keywords else 'None')}
• <b>Replacements:</b> {escape_html(str(len(rule.replace_text)))}
• <b>Triggers:</b> {escape_html(str(rule.trigger_count))}
"""
    if update.callback_query:
        await safe_edit_message(update.callback_query, text, reply_markup=reply_markup, parse_mode=ParseMode.HTML) # Changed to HTML
    else:
        await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=reply_markup) # Changed to HTML
    return EDIT_RULE_SELECT

async def edit_rule_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    
    data = query.data
    rule_id = context.user_data.get('rule_id') or data.split('_')[-1]

    if not rule_id:
        await safe_edit_message(query, get_string(user_lang, "rule_settings_not_found"))
        return ConversationHandler.END

    rule = db.get_rule(rule_id)
    if not rule or rule.user_id != user_id:
        await safe_edit_message(query, get_string(user_lang, "rule_settings_not_found"))
        return ConversationHandler.END

    context.user_data['rule_id'] = rule_id
    context.user_data['current_rule'] = rule

    if data.startswith("edit_rule_toggle_active"):
        rule.is_active = not rule.is_active
        db.save_rule(rule)
        forwarding_engine.reload_rules()
        status = "active" if rule.is_active else "inactive"
        text = get_string(user_lang, "rule_toggle_success").format(rule_id_short=escape_html(rule_id[:8]), status=escape_html(status)) # Changed to HTML
        markup = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data=f"rule_settings_{rule_id}")]])
        await safe_edit_message(query, text, reply_markup=markup, parse_mode=ParseMode.HTML) # Changed to HTML
        return EDIT_RULE_SELECT

    elif data.startswith("edit_rule_delete"):
        keyboard = [
            [InlineKeyboardButton("✅ Yes, Delete", callback_data=f"confirm_delete_rule_{rule_id}")],
            [InlineKeyboardButton("❌ No, Cancel", callback_data=f"rule_settings_{rule_id}")]
        ]
        markup = InlineKeyboardMarkup(keyboard)
        text = get_string(user_lang, "confirm_delete_rule").format(rule_id_short=escape_html(rule_id[:8])) # Changed to HTML
        await safe_edit_message(query, text, reply_markup=markup, parse_mode=ParseMode.HTML) # Changed to HTML
        return EDIT_RULE_SELECT

    elif data.startswith("confirm_delete_rule"):
        db.delete_rule(rule_id)
        forwarding_engine.reload_rules()
        user_profile = db.get_user(user_id)
        if user_profile:
            user_profile.total_rules = max(0, user_profile.total_rules - 1)
            db.save_user(user_profile)
        text = get_string(user_lang, "rule_deleted_success").format(rule_id_short=escape_html(rule_id[:8])) # Changed to HTML
        markup = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Rules", callback_data="my_rules")]])
        await safe_edit_message(query, text, reply_markup=markup, parse_mode=ParseMode.HTML) # Changed to HTML
        context.user_data.clear()
        return ConversationHandler.END

    elif data.startswith("edit_rule_keywords"):
        await safe_edit_message(
            query,
            get_string(user_lang, "rule_edit_keywords").format(
                rule_id_short=escape_html(rule_id[:8]), keywords=escape_html(', '.join(rule.keywords) if rule.keywords else 'None') # Changed to HTML
            ),
            parse_mode=ParseMode.HTML # Changed to HTML
        )
        context.user_data['current_state'] = EDIT_RULE_KEYWORDS # Set state for text input
        return EDIT_RULE_KEYWORDS
    
    elif data.startswith("edit_rule_exclude_keywords"):
        await safe_edit_message(
            query,
            get_string(user_lang, "rule_edit_exclude_keywords").format(
                rule_id_short=escape_html(rule_id[:8]), exclude_keywords=escape_html(', '.join(rule.exclude_keywords) if rule.exclude_keywords else 'None') # Changed to HTML
            ),
            parse_mode=ParseMode.HTML # Changed to HTML
        )
        context.user_data['current_state'] = EDIT_RULE_EXCLUDE_KEYWORDS # Set state for text input
        return EDIT_RULE_EXCLUDE_KEYWORDS

    elif data.startswith("edit_rule_replace_text"):
        replacements_str = "\n".join([f"<code>{escape_html(k)}</code>=<code>{escape_html(v)}</code>" for k, v in rule.replace_text.items()]) if rule.replace_text else "None" # Changed to HTML
        await safe_edit_message(
            query,
            get_string(user_lang, "rule_edit_replace_text").format(
                rule_id_short=escape_html(rule_id[:8]), replace_text=replacements_str # Changed to HTML
            ),
            parse_mode=ParseMode.HTML # Changed to HTML
        )
        context.user_data['current_state'] = EDIT_RULE_REPLACE_TEXT # Set state for text input
        return EDIT_RULE_REPLACE_TEXT

    elif data.startswith("edit_rule_filters"):
        keyboard = [
            [InlineKeyboardButton("🖼️ Allowed Media Types", callback_data=f"edit_rule_filter_media_types_{rule_id}")],
            [InlineKeyboardButton("📏 Message Length", callback_data=f"edit_rule_filter_length_{rule_id}")],
            [InlineKeyboardButton("🔙 Back", callback_data=f"rule_settings_{rule_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await safe_edit_message(
            query,
            get_string(user_lang, "rule_edit_filters").format(rule_id_short=escape_html(rule_id[:8])), # Changed to HTML
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML # Changed to HTML
        )
        return EDIT_RULE_FILTERS

    elif data.startswith("edit_rule_schedule"):
        schedule_status = "Enabled" if rule.schedule.get("enabled") else "Disabled"
        schedule_time = f"{rule.schedule.get('start_time')} - {rule.schedule.get('end_time')}"
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        scheduled_days_names = [day_names[d] for d in rule.schedule.get("days", [])]

        keyboard = [
            [InlineKeyboardButton(f"Toggle Schedule ({schedule_status})", callback_data=f"edit_rule_schedule_toggle_{rule_id}")],
            [InlineKeyboardButton(f"⏰ Time ({schedule_time})", callback_data=f"edit_rule_schedule_time_{rule_id}")],
            [InlineKeyboardButton(f"🗓️ Days ({', '.join(scheduled_days_names)})", callback_data=f"edit_rule_schedule_days_{rule_id}")],
            [InlineKeyboardButton("🔙 Back", callback_data=f"rule_settings_{rule_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await safe_edit_message(
            query,
            get_string(user_lang, "rule_edit_schedule").format(rule_id_short=escape_html(rule_id[:8])), # Changed to HTML
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML # Changed to HTML
        )
        return EDIT_RULE_SCHEDULE

    elif data.startswith("edit_rule_targets"):
        targets_str = ', '.join(f'<code>{escape_html(str(t))}</code>' for t in rule.target_chats) # Changed to HTML
        await safe_edit_message(
            query,
            get_string(user_lang, "rule_edit_targets").format(
                rule_id_short=escape_html(rule_id[:8]), targets=targets_str # Changed to HTML
            ),
            parse_mode=ParseMode.HTML # Changed to HTML
        )
        context.user_data['current_state'] = EDIT_RULE_TARGETS # Set state for text input
        return EDIT_RULE_TARGETS

    elif data.startswith("edit_rule_mode"):
        keyboard = [
            [InlineKeyboardButton("with_tags", callback_data=f"set_mode_with_tags_{rule_id}")],
            [InlineKeyboardButton("without_tags", callback_data=f"set_mode_without_tags_{rule_id}")],
            [InlineKeyboardButton("bypass", callback_data=f"set_mode_bypass_{rule_id}")],
            [InlineKeyboardButton("🔙 Back", callback_data=f"rule_settings_{rule_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await safe_edit_message(
            query,
            get_string(user_lang, "rule_edit_mode").format(
                rule_id_short=escape_html(rule_id[:8]), mode=escape_html(rule.mode) # Changed to HTML
            ),
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=reply_markup
        )
        return EDIT_RULE_MODE
    
    elif data.startswith("set_mode_"):
        new_mode = data.split('_')[2]
        rule.mode = new_mode
        db.save_rule(rule)
        await safe_edit_message(
            query,
            get_string(user_lang, "rule_edit_success").format(rule_id_short=escape_html(rule_id[:8])), # Changed to HTML
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data=f"rule_settings_{rule_id}")]])
        )
        return EDIT_RULE_SELECT

    elif data.startswith("edit_rule_filter_media_types"):
        allowed_types_str = ', '.join(rule.filters.get("allowed_types", [])) if rule.filters.get("allowed_types") else 'None'
        await safe_edit_message(
            query,
            get_string(user_lang, "rule_edit_media_types").format(
                rule_id_short=escape_html(rule_id[:8]), types=escape_html(allowed_types_str) # Changed to HTML
            ),
            parse_mode=ParseMode.HTML # Changed to HTML
        )
        context.user_data['current_state'] = EDIT_RULE_MEDIA_TYPES # Set state for text input
        return EDIT_RULE_MEDIA_TYPES

    elif data.startswith("edit_rule_filter_length"):
        min_len = rule.filters.get("min_length", 0)
        max_len = rule.filters.get("max_length", 0)
        await safe_edit_message(
            query,
            get_string(user_lang, "rule_edit_length_filters").format(
                rule_id_short=escape_html(rule_id[:8]), min_len=escape_html(str(min_len)), max_len=escape_html(str(max_len)) # Changed to HTML
            ),
            parse_mode=ParseMode.HTML # Changed to HTML
        )
        context.user_data['current_state'] = EDIT_RULE_LENGTH_FILTERS # Set state for text input
        return EDIT_RULE_LENGTH_FILTERS

    elif data.startswith("edit_rule_schedule_toggle"):
        rule.schedule["enabled"] = not rule.schedule.get("enabled", False)
        db.save_rule(rule)
        forwarding_engine.reload_rules()
        await safe_edit_message(
            query,
            get_string(user_lang, "rule_edit_success").format(rule_id_short=escape_html(rule_id[:8])), # Changed to HTML
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data=f"edit_rule_schedule_{rule_id}")]])
        )
        return EDIT_RULE_SCHEDULE

    elif data.startswith("edit_rule_schedule_time"):
        start_time = rule.schedule.get("start_time", "00:00")
        end_time = rule.schedule.get("end_time", "23:59")
        await safe_edit_message(
            query,
            get_string(user_lang, "rule_edit_schedule_time").format(
                rule_id_short=escape_html(rule_id[:8]), start_time=escape_html(start_time), end_time=escape_html(end_time) # Changed to HTML
            ),
            parse_mode=ParseMode.HTML # Changed to HTML
        )
        context.user_data['current_state'] = EDIT_RULE_SCHEDULE_TIME # Set state for text input
        return EDIT_RULE_SCHEDULE_TIME

    elif data.startswith("edit_rule_schedule_days"):
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        scheduled_days_names = [day_names[d] for d in rule.schedule.get("days", [])]
        await safe_edit_message(
            query,
            get_string(user_lang, "rule_edit_schedule_days").format(
                rule_id_short=escape_html(rule_id[:8]), days=escape_html(', '.join(scheduled_days_names)) # Changed to HTML
            ),
            parse_mode=ParseMode.HTML # Changed to HTML
        )
        context.user_data['current_state'] = EDIT_RULE_SCHEDULE_DAYS # Set state for text input
        return EDIT_RULE_SCHEDULE_DAYS

    return EDIT_RULE_SELECT # Fallback

async def edit_rule_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    
    rule_id = context.user_data.get('rule_id')
    rule = db.get_rule(rule_id)
    if not rule:
        await update.message.reply_text(get_string(user_lang, "rule_settings_not_found"))
        return ConversationHandler.END

    user_input = update.message.text.strip()
    current_state = context.user_data['current_state'] # Store current state in user_data to know what to update

    try:
        if current_state == EDIT_RULE_KEYWORDS:
            rule.keywords = [k.strip() for k in user_input.split(',') if k.strip()]
        elif current_state == EDIT_RULE_EXCLUDE_KEYWORDS:
            rule.exclude_keywords = [k.strip() for k in user_input.split(',') if k.strip()]
        elif current_state == EDIT_RULE_REPLACE_TEXT:
            replacements = {}
            for line in user_input.split('\n'):
                if '=' in line:
                    old, new = line.split('=', 1)
                    replacements[old.strip()] = new.strip()
            rule.replace_text = replacements
        elif current_state == EDIT_RULE_TARGETS:
            target_chats = [validate_chat_id(chat.strip()) for chat in user_input.split(',') if chat.strip()]
            rule.target_chats = target_chats
        elif current_state == EDIT_RULE_MEDIA_TYPES:
            allowed_types = [t.strip().lower() for t in user_input.split(',') if t.strip()]
            valid_types = ["text", "photo", "video", "document", "audio", "voice", "sticker", "animation", "poll", "contact", "location", "venue", "video_note"]
            rule.filters["allowed_types"] = [t for t in allowed_types if t in valid_types]
        elif current_state == EDIT_RULE_LENGTH_FILTERS:
            min_len, max_len = map(int, user_input.split(','))
            rule.filters["min_length"] = min_len
            rule.filters["max_length"] = max_len
        elif current_state == EDIT_RULE_SCHEDULE_TIME:
            start_time_str, end_time_str = user_input.split('-')
            datetime.strptime(start_time_str.strip(), "%H:%M") # Validate format
            datetime.strptime(end_time_str.strip(), "%H:%M") # Validate format
            rule.schedule["start_time"] = start_time_str.strip()
            rule.schedule["end_time"] = end_time_str.strip()
        elif current_state == EDIT_RULE_SCHEDULE_DAYS:
            day_map = {"mon":0, "tue":1, "wed":2, "thu":3, "fri":4, "sat":5, "sun":6}
            days_input = [d.strip().lower() for d in user_input.split(',') if d.strip()]
            rule.schedule["days"] = sorted([day_map[d] for d in days_input if d in day_map])
        else:
            await update.message.reply_text(get_string(user_lang, "invalid_input"))
            return current_state # Stay in current state

        db.save_rule(rule)
        forwarding_engine.reload_rules()
        await update.message.reply_text(
            get_string(user_lang, "rule_edit_success").format(rule_id_short=escape_html(rule_id[:8])), # Changed to HTML
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data=f"rule_settings_{rule_id}")]])
        )
        return EDIT_RULE_SELECT # Go back to rule settings menu
    except Exception as e:
        await update.message.reply_text(f"{get_string(user_lang, 'invalid_input')}\nError: {escape_html(str(e))}") # Changed to HTML
        return current_state # Stay in current state

# --- Broadcast Conversation ---

async def broadcast_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    
    # Use effective_message for consistency
    message_to_reply = update.effective_message

    # Check if command was sent with arguments (direct message broadcast)
    if context.args and len(context.args) > 0:
        message_text = " ".join(context.args)
        await send_broadcast(update, context, message_text, "all")
        return ConversationHandler.END

    keyboard = [
        [InlineKeyboardButton("Send to All Users", callback_data="broadcast_all")],
        [InlineKeyboardButton("Send to Premium Users", callback_data="broadcast_role_premium")],
        [InlineKeyboardButton("Send to Admin Users", callback_data="broadcast_role_admin")],
        [InlineKeyboardButton(get_string(user_lang, "cancel_operation"), callback_data="cancel_conversation")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await message_to_reply.reply_text(
        "📢 <b>Broadcast System</b>\n\n" # Changed to HTML
        "Select target audience or send message directly with <code>/broadcast &lt;message&gt;</code>:", # Changed to HTML
        parse_mode=ParseMode.HTML, # Changed to HTML
        reply_markup=reply_markup
    )
    return BROADCAST_MESSAGE_STATE


async def broadcast_select_target(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'

    if query.data == "cancel_conversation":
        return await cancel_conversation(update, context)

    target_type = query.data.replace("broadcast_", "")
    context.user_data['broadcast_target_type'] = target_type

    await safe_edit_message(
        query,
        f"Selected target: <b>{escape_html(target_type.replace('_', ' ').title())}</b>\n\n" # Changed to HTML
        "Please send the message you want to broadcast.", # Changed to HTML
        parse_mode=ParseMode.HTML # Changed to HTML
    )
    return BROADCAST_ROLE_MESSAGE_STATE


async def broadcast_send_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    message_text = update.message.text
    target_type = context.user_data.get('broadcast_target_type', 'all')

    await send_broadcast(update, context, message_text, target_type)

    context.user_data.clear()
    return ConversationHandler.END


async def send_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE, message_text: str, target_type: str):
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'

    all_users = db.get_all_users()
    if target_type == "all":
        target_users = all_users
    elif target_type.startswith("role_"):
        role = target_type.replace("role_", "")
        target_users = [u for u in all_users if u.role == role]
    else:
        target_users = []

    sent_count = 0
    failed_count = 0

    try:
        # Use effective_message for consistency
        status_message = await update.effective_message.reply_text("Sending broadcast... 0/0")
    except Exception:
        status_message = None

    broadcast_template = GLOBAL_SETTINGS.get(
        "broadcast_template",
        "📢 <b>Broadcast Message</b>\n\n{message}\n\n<i>Sent by bot admin</i>" # Changed to HTML
    )
    # Ensure broadcast template is HTML escaped
    final_message = broadcast_template.format(message=escape_html(message_text)) # Changed to HTML

    for user_profile in target_users:
        try:
            await context.bot.send_message(
                chat_id=user_profile.user_id,
                text=final_message,
                parse_mode=ParseMode.HTML # Changed to HTML
            )
            sent_count += 1
        except Forbidden:
            logger.warning(f"Broadcast failed for user {user_profile.user_id}: Bot blocked by user.")
            failed_count += 1
        except TelegramError as e:
            logger.error(f"Broadcast failed for user {user_profile.user_id}: {e}")
            failed_count += 1

        if status_message and ((sent_count + failed_count) % 10 == 0 or (sent_count + failed_count) == len(target_users)):
            try:
                await status_message.edit_text(f"Sending broadcast... {sent_count}/{len(target_users)} (Failed: {failed_count})")
            except Exception:
                pass

        await asyncio.sleep(0.1)

    if status_message:
        await status_message.edit_text(
            f"✅ Broadcast complete!\n\nSent to {sent_count} users. Failed for {failed_count} users.", # Changed to HTML
            parse_mode=ParseMode.HTML # Changed to HTML
        )

    db.log_action(
        user_id, 'broadcast',
        f"Sent to {target_type} users",
        status='success',
        details=f"Sent: {sent_count}, Failed: {failed_count}"
    )
broadcast_handler = ConversationHandler(
    entry_points=[
        CommandHandler("broadcast", broadcast_start)
    ],
    states={
        BROADCAST_MESSAGE_STATE: [ # Use the renamed state
            CallbackQueryHandler(broadcast_select_target, pattern="^broadcast_.*")
        ],
        BROADCAST_ROLE_MESSAGE_STATE: [ # Use the renamed state
            MessageHandler(filters.TEXT & ~filters.COMMAND, broadcast_send_message)
        ]
    },
    fallbacks=[
        CallbackQueryHandler(cancel_conversation, pattern="^cancel_conversation$"),
        CommandHandler("cancel", cancel_conversation)
    ]
)

# ============================================================================
# COMMAND HANDLERS
# ============================================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    user = update.effective_user
    user_id = user.id
    user_lang = 'en'  # Default language for new users

    # Create or update user profile
    user_profile = db.get_user(user_id)
    if not user_profile:
        user_profile = UserProfile(
            user_id=user_id,
            username=user.username or "",
            first_name=user.first_name or "",
            last_name=user.last_name or "",
            joined_date=get_current_time(),
            last_active=get_current_time()
        )
        if user_id == OWNER_ID:
            user_profile.role = 'owner'
        db.save_user(user_profile)
        today = datetime.now().strftime('%Y-%m-%d')
        db.update_daily_statistics(today, {}, {'users_joined': 1})
    else:
        user_profile.last_active = get_current_time()
        if user.username:
            user_profile.username = user.username
        if user.first_name:
            user_profile.first_name = user.first_name
        if user.last_name:
            user_profile.last_name = user.last_name
        if user_id == OWNER_ID and user_profile.role != 'owner':
            user_profile.role = 'owner'
        db.save_user(user_profile)
        user_lang = user_profile.settings.get('language', 'en')

    role = RoleManager.get_role(user_id)
    phone_status = "✅ Not verified" # Default to not verified
    if PhoneVerification.has_phone(user_id):
        phone_status = "✅ Verified"

    # Escape dynamic content for HTML
    escaped_first_name = escape_html(user.first_name)
    escaped_role = escape_html(role.title())
    escaped_phone_status = escape_html(phone_status)

    welcome_text = (
        "🚀 <b>Welcome to ForwardBot Premium!</b>\n\n"
        f"👤 <b>User:</b> {escaped_first_name}\n"
        f"🎭 <b>Role:</b> {escaped_role}\n"
        f"📱 <b>Phone:</b> {escaped_phone_status}\n\n"
        "<b>🎯 Three Forwarding Modes:</b>\n"
        "• <code>with_tags</code> - Keep original sender info\n"
        "• <code>without_tags</code> - Clean forwarding\n"
        "• <code>bypass</code> - Bypass restrictions\n\n"
        "<b>🔥 Premium Features:</b>\n"
        "• Auto-forwarding rules\n"
        "• Keyword filtering\n"
        "• Message scheduling\n"
        "• Advanced statistics\n"
        "• Phone verification\n"
        "• Batch operations\n"
        "• Custom templates\n\n"
        "Use /help to see all commands!"
    )

    keyboard = [
        [
            InlineKeyboardButton("📚 Help", callback_data="help"),
            InlineKeyboardButton("⚡ Quick Start", callback_data="quickstart")
        ],
        [
            InlineKeyboardButton("📱 Connect Phone", callback_data="connect_phone"),
            InlineKeyboardButton("📊 My Stats", callback_data="user_stats")
        ]
    ]

    if RoleManager.is_admin(user_id):
        keyboard.append([
            InlineKeyboardButton("⚙️ Admin Panel", callback_data="admin_panel")
        ])

    reply_markup = InlineKeyboardMarkup(keyboard)

    # Send welcome video
    # Ensure this path is correct and accessible
    try:
        await context.bot.send_video(
            chat_id=update.effective_chat.id,
            video="https://files.catbox.moe/kg7jcs.mp4" # Example URL, replace with your actual video file_id or URL
        )
    except Exception as e:
        logger.error(f"Failed to send welcome video: {e}")
        # Continue without video if it fails

    # Send welcome message
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=welcome_text,
        parse_mode=ParseMode.HTML, # Changed to HTML
        reply_markup=reply_markup
    )


# --- HELP COMMAND ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    role = RoleManager.get_role(user_id)

    help_text = (
        "📚 <b>ForwardBot Help</b>\n\n"
        "<b>🔰 Basic Commands:</b>\n"
        "/start - Welcome message and bot info\n"
        "/help - Show this help message\n"
        "/status - Check bot status and your info\n"
        "/connect_phone - Verify your phone number\n"
        "/settings - Manage your personal settings\n\n"

        "<b>⚡ Forwarding Commands:</b>\n"
        "/forward - One-time forward (reply to a message)\n"
        "/autoforward - Create auto-forwarding rule\n"
        "/rules - Manage your forwarding rules\n\n"

        "<b>📋 Forwarding Modes:</b>\n"
        "• <code>with_tags</code> - Preserve original message info\n"
        "• <code>without_tags</code> - Clean forwarding (copy)\n"
        "• <code>bypass</code> - Bypass restrictions + clean\n\n"

        "<b>📊 Statistics:</b>\n"
        "/stats - View your forwarding statistics\n"
        "/logs - View recent activity logs\n\n"
    )

    if role in ['premium', 'admin', 'owner']:
        help_text += (
            "<b>💎 Premium Commands:</b>\n"
            "/batch_forward - Forward multiple messages\n"
            "/export - Export your data\n\n"
        )

    if role in ['admin', 'owner']:
        help_text += (
            "<b>👑 Admin Commands:</b>\n"
            "/users - Manage users\n"
            "/promote - Promote user\n"
            "/demote - Demote user\n"
            "/ban - Ban user\n"
            "/unban - Unban user\n"
            "/broadcast - Send broadcast message\n"
            "/system_stats - System statistics\n"
            "/bot_settings - Configure global bot settings\n\n"
        )

    if role == 'owner':
        help_text += (
            "<b>🔧 Owner Commands:</b>\n"
            "/maintenance - Toggle maintenance mode\n"
            "/backup - Create system backup\n"
            "/restore - Restore from backup\n"
            "/debug - Debug information\n"
            "/clear_logs - Clear old logs\n\n"
        )

    help_text += (
        "<b>💡 Tips:</b>\n"
        "• Use negative chat IDs for channels/groups\n"
        "• Premium features require verification\n"
        "• Contact admin for role upgrades\n\n"
        "<b>🆘 Support:</b>\n"
        "Contact @MD_TECH_bot for assistance"
    )

    keyboard = [
        [
            InlineKeyboardButton("🚀 Quick Start", callback_data="quickstart"),
            InlineKeyboardButton("💎 Premium Info", callback_data="premium_info")
        ],
        [
            InlineKeyboardButton("📱 Phone Setup", callback_data="connect_phone"),
            InlineKeyboardButton("⚡ Examples", callback_data="examples")
        ]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.message:
        await update.message.reply_text(
            help_text,
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=reply_markup
        )
    elif update.callback_query:
        await safe_edit_message(update.callback_query, text=help_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML) # Changed to HTML


# --- STATUS COMMAND ---
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    role = RoleManager.get_role(user_id)
    phone_verified = PhoneVerification.has_phone(user_id)
    user_profile = db.get_user(user_id)
    user_rules = db.get_rules(user_id)
    active_rules = len([r for r in user_rules if r.is_active])
    total_users = len(db.get_all_users())
    system_health = "🟢 Healthy"
    uptime_seconds = int(time.time() - context.bot_data.get('start_time', time.time()))

    status_text = (
        "📊 <b>Bot Status</b>\n\n"
        f"<b>👤 Your Profile:</b>\n"
        f"• Name: {escape_html(user.first_name)}\n"
        f"• Role: {escape_html(role.title())}\n"
        f"• Phone: {escape_html('✅ Verified' if phone_verified else '❌ Not verified')}\n"
        f"• Rules: {active_rules} active, {len(user_rules)} total\n"
        f"• Forwards: {user_profile.total_forwards if user_profile else 0}\n\n"
        f"<b>🤖 System Status:</b>\n"
        f"• Health: {escape_html(system_health)}\n"
        f"• Total Users: {total_users}\n"
        f"• Uptime: {escape_html(format_duration(uptime_seconds))}\n"
        f"• Version: 3.0.0-premium\n\n"
        f"<b>⚙️ Settings:</b>\n"
        f"• Auto Forward: {'✅' if GLOBAL_SETTINGS['auto_forward'] else '❌'}\n"
        f"• Phone Required: {'✅' if PhoneVerification.is_required() else '❌'}\n"
        f"• Rate Limit: {GLOBAL_SETTINGS['rate_limit']}/min\n"
        f"• Maintenance: {'✅' if GLOBAL_SETTINGS.get('maintenance_mode') else '❌'}"
    )

    keyboard = [
        [InlineKeyboardButton("🔄 Refresh", callback_data="refresh_status"),
         InlineKeyboardButton("📊 Detailed Stats", callback_data="detailed_stats")],
        [InlineKeyboardButton("⚙️ My Settings", callback_data="user_settings"),
         InlineKeyboardButton("📝 My Rules", callback_data="my_rules")]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.message:
      await update.message.reply_text(status_text, parse_mode=ParseMode.HTML, reply_markup=reply_markup) # Changed to HTML
    elif update.callback_query:
      await safe_edit_message(update.callback_query, status_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML) # Changed to HTML


# --- CONNECT PHONE COMMAND ---
async def connect_phone_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    user_profile = db.get_user(user_id)

    # Use effective_message for consistency
    message_to_reply = update.effective_message

    if user_profile and user_profile.phone:
        if update.callback_query:
            await update.callback_query.answer(get_string(user_lang, "phone_already_verified"), show_alert=True)
        else:
            await message_to_reply.reply_text(get_string(user_lang, "phone_already_verified"), reply_markup=ReplyKeyboardRemove())
        return

    keyboard = [
        [KeyboardButton(get_string(user_lang, "Share Phone Number"), request_contact=True)],
        [KeyboardButton(get_string(user_lang, "cancel_operation"))]
    ]
    text = get_string(user_lang, "phone_verification_text")
    markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

    if update.callback_query:
        await update.callback_query.answer()
        await context.bot.send_message(chat_id=update.effective_chat.id, text=text, parse_mode=ParseMode.HTML, reply_markup=markup) # Changed to HTML
        try:
            # Attempt to remove the inline keyboard from the original message if it was a callback
            await update.callback_query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
    else:
        await message_to_reply.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=markup) # Changed to HTML


# --- CONTACT HANDLER ---
async def contact_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    contact = update.message.contact
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'

    if contact.user_id != user_id:
        await update.message.reply_text(get_string(user_lang, "share_own_phone"), reply_markup=ReplyKeyboardRemove())
        return

    phone_number = f"+{contact.phone_number.lstrip('+')}"
    PhoneVerification.save_phone(user_id, phone_number)

    confirmation_text = (
        f"{get_string(user_lang, 'phone_verified_success')}\n"
        f"<b>📱 Number:</b> <code>{escape_html(phone_number)}</code>\n" # Changed to HTML
        f"<b>👤 Name:</b> {escape_html(contact.first_name)} {escape_html(contact.last_name or '')}\n\n" # Changed to HTML
        f"🎉 You can now use all premium forwarding features!"
    )

    await update.message.reply_text(confirmation_text, parse_mode=ParseMode.HTML, reply_markup=ReplyKeyboardRemove()) # Changed to HTML

    db.log_action(user_id=user_id, action='phone_verification', details=f"Phone: {phone_number}", status='success')

@premium_required
@rate_limit(max_calls=10, time_window=60)
async def forward_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Forward command handler - forwards a replied-to message"""
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    
    if not PhoneVerification.check_access(user_id):
        await update.message.reply_text(get_string(user_lang, "phone_verification_required"))
        return
    
    if not update.message.reply_to_message:
        await update.message.reply_text(
            get_string(user_lang, "forward_command_usage") + "\n\n"
            "To use <code>/forward</code>, <b>reply to the message</b> you want to forward.", # Changed to HTML
            parse_mode=ParseMode.HTML # Changed to HTML
        )
        return
    
    if len(context.args) < 2:
        await update.message.reply_text(
            get_string(user_lang, "forward_command_usage") + "\n\n"
            "Example: <code>/forward without_tags -1009876543210</code> (reply to message)", # Changed to HTML
            parse_mode=ParseMode.HTML # Changed to HTML
        )
        return
    
    mode = context.args[0]
    if mode not in ['with_tags', 'without_tags', 'bypass']:
        await update.message.reply_text(get_string(user_lang, "invalid_mode"), parse_mode=ParseMode.HTML) # Changed to HTML
        return
    
    try:
        target_chats = [validate_chat_id(chat) for chat in context.args[1:]]
    except ValueError as e:
        await update.message.reply_text(get_string(user_lang, "invalid_chat_id_format").format(error=escape_html(str(e))), parse_mode=ParseMode.HTML) # Changed to HTML
        return
    
    message_to_forward = update.message.reply_to_message
    
    processing_msg = await update.message.reply_text(get_string(user_lang, "forward_processing"))
    
    try:
        stats = await forwarding_engine.manual_forward_message(
            bot=context.bot,
            message=message_to_forward,
            mode=mode,
            target_chats=target_chats,
            user_id=user_id
        )
        
        result_text = get_string(user_lang, "forward_complete") + f"""

📊 <b>Results:</b>
• Successful: {stats['successful']}
• Failed: {stats['failed']}
• Mode: {escape_html(mode)}

📤 <b>Source:</b> <code>{escape_html(str(message_to_forward.chat.id))}</code>
🎯 <b>Targets:</b> {', '.join(f'<code>{escape_html(str(t))}</code>' for t in target_chats)}
"""
        
        if stats['errors']:
            result_text += f"\n❌ <b>Errors:</b>\n" + '\n'.join(f"• {escape_html(error)}" for error in stats['errors'][:5]) # Changed to HTML
        
        await processing_msg.edit_text(result_text, parse_mode=ParseMode.HTML) # Changed to HTML
        
        user_profile = db.get_user(user_id)
        if user_profile:
            user_profile.total_forwards += stats['successful']
            db.save_user(user_profile)
    
    except Exception as e:
        logger.error(f"Forward command error: {e}")
        await processing_msg.edit_text(
            f"{get_string(user_lang, 'forward_failed')}\n\nError: {escape_html(str(e))}", # Changed to HTML
            parse_mode=ParseMode.HTML # Changed to HTML
        )

@premium_required
async def autoforward_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Auto-forward command handler - initiates conversation for rule creation"""
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'

    if not PhoneVerification.check_access(user_id):
        # Use effective_message for consistency
        message_to_reply = update.effective_message
        if message_to_reply:
            await message_to_reply.reply_text(get_string(user_lang, "phone_verification_required"))
        return ConversationHandler.END # End if no access

    # Start the conversation for rule creation
    return await create_rule_start(update, context)

async def rules_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Rules management command"""
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    user_rules = db.get_rules(user_id)

    if update.callback_query:
        message_to_edit = update.callback_query.message
    else:
        message_to_edit = update.message

    if not message_to_edit:
        return

    if not user_rules:
        text = get_string(user_lang, "no_rules_found")
        if update.callback_query:
            await safe_edit_message(update.callback_query, text, parse_mode=ParseMode.HTML) # Changed to HTML
        else:
            await message_to_edit.reply_text(text, parse_mode=ParseMode.HTML) # Changed to HTML
        return

    rules_text = f"<b>{escape_html(get_string(user_lang, 'your_rules').format(count=len(user_rules)))}</b>\n\n" # Changed to HTML

    for i, rule in enumerate(user_rules[:10], 1):
        status = "🟢 Active" if rule.is_active else "🔴 Inactive"
        target_count = len(rule.target_chats)

        rules_text += (
            f"<b>{i}. Rule</b> <code>{escape_html(str(rule.rule_id[:8]))}</code>...\n" # Changed to HTML
            f"• Status: {escape_html(status)}\n" # Changed to HTML
            f"• Source: <code>{escape_html(str(rule.source_chat))}</code>\n" # Changed to HTML
            f"• Targets: {target_count} chat{'s' if target_count != 1 else ''}\n"
            f"• Mode: {escape_html(str(rule.mode))}\n" # Changed to HTML
            f"• Triggers: {rule.trigger_count}\n\n"
        )

    if len(user_rules) > 10:
        rules_text += f"\n... and {len(user_rules) - 10} more rules. Use 'Manage Rules' to see all."

    keyboard = [
        [
            InlineKeyboardButton("➕ New Rule", callback_data="new_rule"),
            InlineKeyboardButton("🔄 Refresh", callback_data="refresh_rules")
        ],
        [
            InlineKeyboardButton("⚙️ Manage Rules", callback_data="manage_all_rules"),
            InlineKeyboardButton("📊 Rule Stats", callback_data="rule_stats_overview")
        ]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.callback_query:
        await safe_edit_message(
            update.callback_query,
            rules_text,
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=reply_markup
        )
    else:
        await message_to_edit.reply_text(
            rules_text,
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=reply_markup
        )

async def manage_all_rules_command(update: Update, context: ContextTypes.DEFAULT_TYPE, page: int = 0):
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    user_rules = db.get_rules(user_id)
    
    if not user_rules:
        await safe_edit_message(update.callback_query, get_string(user_lang, "no_rules_found"), parse_mode=ParseMode.HTML) # Changed to HTML
        return

    rules_per_page = 5
    total_pages = (len(user_rules) + rules_per_page - 1) // rules_per_page
    start_index = page * rules_per_page
    end_index = start_index + rules_per_page
    
    paginated_rules = user_rules[start_index:end_index]

    text = get_string(user_lang, "your_rules").format(count=len(user_rules)) + f" (Page {page + 1}/{total_pages})\n\n"
    
    rule_buttons = []
    for rule in paginated_rules:
        status_emoji = "🟢" if rule.is_active else "🔴"
        rule_buttons.append([InlineKeyboardButton(f"{status_emoji} Rule {escape_html(rule.rule_id[:8])}... (Source: {escape_html(str(rule.source_chat))})", callback_data=f"rule_settings_{rule.rule_id}")]) # Changed to HTML
    
    pagination_buttons = []
    if page > 0:
        pagination_buttons.append(InlineKeyboardButton("⬅️ Prev", callback_data=f"manage_all_rules_page_{page - 1}"))
    if page < total_pages - 1:
        pagination_buttons.append(InlineKeyboardButton("Next ➡️", callback_data=f"manage_all_rules_page_{page + 1}"))

    keyboard = rule_buttons + [pagination_buttons] + [[InlineKeyboardButton("🔙 Back to Rules Menu", callback_data="my_rules")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await safe_edit_message(
        update.callback_query,
        text,
        parse_mode=ParseMode.HTML, # Changed to HTML
        reply_markup=reply_markup
    )

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Statistics command handler"""
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    user_profile = db.get_user(user_id)
    
    if update.callback_query:
        message_to_edit = update.callback_query.message
    else:
        message_to_edit = update.message

    if not message_to_edit:
        return

    if not user_profile:
        if update.callback_query:
            await safe_edit_message(update.callback_query, get_string(user_lang, "no_stats_available"), parse_mode=ParseMode.HTML) # Changed to HTML
        else:
            await message_to_edit.reply_text(get_string(user_lang, "no_stats_available"), parse_mode=ParseMode.HTML) # Changed to HTML
        return
    
    user_rules = db.get_rules(user_id)
    active_rules = len([r for r in user_rules if r.is_active])
    total_triggers = sum(r.trigger_count for r in user_rules)
    
    success_rate = 0
    if user_profile.total_forwards > 0:
        success_rate = (user_profile.total_forwards / max(1, total_triggers)) * 100 if total_triggers > 0 else 0
        success_rate = min(100, success_rate)
    
    stats_text = get_string(user_lang, "your_stats") + f"""

<b>👤 Profile:</b>
• Name: {escape_html(user_profile.first_name)}
• Role: {escape_html(user_profile.role.title())}
• Member since: {escape_html(datetime.fromisoformat(user_profile.joined_date).strftime('%Y-%m-%d'))}
• Last active: {escape_html(datetime.fromisoformat(user_profile.last_active).strftime('%Y-%m-%d %H:%M'))}

<b>📈 Forwarding Stats:</b>
• Total forwards: {user_profile.total_forwards:,}
• Success rate: {success_rate:.1f}%
• Total rules: {len(user_rules)}
• Active rules: {active_rules}
• Rule triggers: {total_triggers:,}

<b>🎯 Performance:</b>
• Avg. forwards/rule: {user_profile.total_forwards / max(1, len(user_rules)):.1f}
• Most active rule: {max(user_rules, key=lambda r: r.trigger_count).trigger_count if user_rules else 0} triggers
• Phone verified: {'✅' if user_profile.phone else '❌'}
"""
    
    keyboard = [
        [
            InlineKeyboardButton("📊 Detailed Stats", callback_data="detailed_user_stats"),
            InlineKeyboardButton("📈 Charts", callback_data="stats_charts")
        ],
        [
            InlineKeyboardButton("📝 Export Data", callback_data="export_user_data"),
            InlineKeyboardButton("🔄 Refresh", callback_data="refresh_user_stats")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.callback_query:
        await safe_edit_message(
            update.callback_query,
            stats_text,
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=reply_markup
        )
    else:
        await message_to_edit.reply_text(
            stats_text,
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=reply_markup
        )

async def logs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    
    logs = db.get_logs(user_id=user_id, limit=20)
    
    if not logs:
        text = "📋 <b>Your Recent Activity Logs</b>\n\nNo recent activity found." # Changed to HTML
    else:
        text = "📋 <b>Your Recent Activity Logs</b>\n\n" # Changed to HTML
        for log in logs:
            timestamp = datetime.fromisoformat(log['timestamp']).strftime('%Y-%m-%d %H:%M')
            status_emoji = "✅" if log['status'] == 'success' else "❌"
            text += f"• {escape_html(timestamp)} {status_emoji} <b>{escape_html(log['action'].replace('_', ' ').title())}</b>: {escape_html(log['details'])}\n" # Changed to HTML
            if log['error_message']:
                text += f"  <i>Error: {escape_html(truncate_text(log['error_message'], 50))}</i>\n" # Changed to HTML
    
    keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="refresh_logs")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.message:
        await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=reply_markup) # Changed to HTML
    elif update.callback_query:
        await safe_edit_message(update.callback_query, text, parse_mode=ParseMode.HTML, reply_markup=reply_markup) # Changed to HTML

async def user_settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_profile = db.get_user(user_id)
    user_lang = user_profile.settings.get('language', 'en')

    settings = user_profile.settings

    text = f"⚙️ <b>Your Personal Settings</b>\n\n" \
           f"• Notifications: {'✅ Enabled' if settings.get('notifications', True) else '❌ Disabled'}\n" \
           f"• Auto-Delete: {'✅ Enabled' if settings.get('auto_delete', False) else '❌ Disabled'}\n" \
           f"• Language: {escape_html(settings.get('language', 'en').upper())}\n" \
           f"• Timezone: {escape_html(settings.get('timezone', 'UTC'))}\n" \
           f"• Privacy Mode: {escape_html(settings.get('privacy_mode', 'default').title())}"

    keyboard = [
        [InlineKeyboardButton(f"Toggle Notifications ({'ON' if settings.get('notifications', True) else 'OFF'})", callback_data="toggle_user_setting_notifications")],
        [InlineKeyboardButton(f"Toggle Auto-Delete ({'ON' if settings.get('auto_delete', False) else 'OFF'})", callback_data="toggle_user_setting_auto_delete")],
        [InlineKeyboardButton(f"Change Language ({settings.get('language', 'en').upper()})", callback_data="change_user_setting_language")],
        [InlineKeyboardButton(f"Change Timezone ({settings.get('timezone', 'UTC')})", callback_data="change_user_setting_timezone")],
        [InlineKeyboardButton(f"Change Privacy Mode ({settings.get('privacy_mode', 'default').title()})", callback_data="change_user_setting_privacy_mode")],
        [InlineKeyboardButton("🔙 Back to Status", callback_data="refresh_status")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.message:
        await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=reply_markup) # Changed to HTML
    elif update.callback_query:
        await safe_edit_message(update.callback_query, text, parse_mode=ParseMode.HTML, reply_markup=reply_markup) # Changed to HTML

# ============================================================================
# ADMIN COMMANDS
# ============================================================================

@admin_required
async def users_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Users management command (admin only)"""
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    
    all_users = db.get_all_users()
    total_users = len(all_users)
    premium_users = len([u for u in all_users if u.role == 'premium'])
    admin_users = len([u for u in all_users if u.role == 'admin'])
    banned_users = len([u for u in all_users if u.is_banned])
    
    users_text = get_string(user_lang, "users_management") + f"""

<b>📊 Overview:</b>
• Total Users: {total_users:,}
• Premium Users: {premium_users}
• Admin Users: {admin_users}
• Free Users: {total_users - premium_users - admin_users}
• Banned Users: {banned_users}

<b>🚀 Recent Activity:</b>
• New users today: {db.get_statistics().daily_stats.get(datetime.now().strftime('%Y-%m-%d'), {}).get('users_joined', 0)}
• Total rules created today: {db.get_statistics().daily_stats.get(datetime.now().strftime('%Y-%m-%d'), {}).get('total_rules_created', 0)}
• Total forwards today: {db.get_statistics().daily_stats.get(datetime.now().strftime('%Y-%m-%d'), {}).get('total_forwards', 0)}
"""
    
    keyboard = [
        [
            InlineKeyboardButton("👤 User Search", callback_data="user_search"),
            InlineKeyboardButton("📊 User Stats", callback_data="admin_user_stats")
        ],
        [
            InlineKeyboardButton("🎖️ Promote User", callback_data="promote_user_menu"),
            InlineKeyboardButton("📉 Demote User", callback_data="demote_user_menu")
        ],
        [
            InlineKeyboardButton("🚫 Ban User", callback_data="ban_user_menu"),
            InlineKeyboardButton("✅ Unban User", callback_data="unban_user_menu")
        ],
        [
            InlineKeyboardButton("📢 Broadcast", callback_data="broadcast_menu")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.message:
        await update.message.reply_text(
            users_text,
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=reply_markup
        )
    elif update.callback_query:
        await safe_edit_message(
            update.callback_query,
            users_text,
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=reply_markup
        )

@admin_required
async def promote_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Promote user command (admin only)"""
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'

    if len(context.args) != 2:
        await update.message.reply_text(get_string(user_lang, "promote_command_usage"), parse_mode=ParseMode.HTML) # Changed to HTML
        return
    
    try:
        target_user_id = int(context.args[0])
        role = context.args[1].lower()
    except ValueError:
        await update.message.reply_text(get_string(user_lang, "invalid_user_id_format"))
        return
    
    if role not in ['premium', 'admin']:
        await update.message.reply_text(get_string(user_lang, "invalid_role"))
        return
    
    if role == 'admin' and not RoleManager.is_owner(update.effective_user.id):
        await update.message.reply_text(get_string(user_lang, "owner_only_admin_promote"))
        return
    
    if RoleManager.promote_user(target_user_id, role):
        await update.message.reply_text(
            get_string(user_lang, "user_promoted_success") + f"""
👤 <b>User ID:</b> <code>{target_user_id}</code>
🎖️ <b>New Role:</b> <b>{escape_html(role.title())}</b>

The user has been notified of their promotion.
""",
            parse_mode=ParseMode.HTML # Changed to HTML
        )
        
        try:
            await context.bot.send_message(
                chat_id=target_user_id,
                text=f"🎉 <b>Congratulations!</b> You have been promoted to <b>{escape_html(role.title())}</b> role!\nUse /help to see your new privileges.", # Changed to HTML
                parse_mode=ParseMode.HTML # Changed to HTML
            )
        except Exception:
            pass
        
        db.log_action(
            user_id=update.effective_user.id,
            action='promote_user',
            details=f"Promoted {target_user_id} to {role}",
            status='success'
        )
    else:
        await update.message.reply_text(get_string(user_lang, "failed_to_promote_user"))

@admin_required
async def demote_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'

    if len(context.args) < 1 or len(context.args) > 2:
        await update.message.reply_text(
            "📝 <b>Demote User Usage:</b>\n\n" # Changed to HTML
            "<code>/demote &lt;user_id&gt; [role]</code>\n\n" # Changed to HTML
            "Default role is <code>free</code>. Available roles: <code>free</code>, <code>premium</code>\n\n" # Changed to HTML
            "Example: <code>/demote 123456789 free</code>", # Changed to HTML
            parse_mode=ParseMode.HTML # Changed to HTML
        )
        return
    
    try:
        target_user_id = int(context.args[0])
        role = context.args[1].lower() if len(context.args) == 2 else 'free'
    except ValueError:
        await update.message.reply_text(get_string(user_lang, "invalid_user_id_format"))
        return
    
    if role not in ['free', 'premium']:
        await update.message.reply_text("❌ Invalid demotion role. Use 'free' or 'premium'.")
        return

    if RoleManager.demote_user(target_user_id, role):
        await update.message.reply_text(
            f"✅ <b>User Demoted!</b>\n\n" # Changed to HTML
            f"👤 <b>User ID:</b> <code>{target_user_id}</code>\n" # Changed to HTML
            f"📉 <b>New Role:</b> <b>{escape_html(role.title())}</b>", # Changed to HTML
            parse_mode=ParseMode.HTML # Changed to HTML
        )
        try:
            await context.bot.send_message(
                chat_id=target_user_id,
                text=f"⚠️ You have been demoted to <b>{escape_html(role.title())}</b> role.", # Changed to HTML
                parse_mode=ParseMode.HTML # Changed to HTML
            )
        except Exception:
            pass
        db.log_action(user_id, 'demote_user', f"Demoted {target_user_id} to {role}", status='success')
    else:
        await update.message.reply_text("❌ Failed to demote user.")

@admin_required
async def ban_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'

    if len(context.args) < 1:
        await update.message.reply_text(get_string(user_lang, "ban_command_usage"), parse_mode=ParseMode.HTML) # Changed to HTML
        return ConversationHandler.END
    
    try:
        target_user_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text(get_string(user_lang, "invalid_user_id_format"))
        return ConversationHandler.END

    if target_user_id == OWNER_ID:
        await update.message.reply_text("❌ Cannot ban the owner.")
        return ConversationHandler.END
    
    user_profile = db.get_user(target_user_id)
    if not user_profile:
        await update.message.reply_text(get_string(user_lang, "user_not_found"))
        return ConversationHandler.END

    if user_profile.is_banned:
        await update.message.reply_text(f"❌ User {target_user_id} is already banned.")
        return ConversationHandler.END

    reason = " ".join(context.args[1:]) if len(context.args) > 1 else "No reason provided."
    
    user_profile.is_banned = True
    user_profile.ban_reason = reason
    db.save_user(user_profile)

    await update.message.reply_text(
        get_string(user_lang, "user_banned_success").format(user_id=target_user_id) + f"\nReason: {escape_html(reason)}", # Changed to HTML
        parse_mode=ParseMode.HTML # Changed to HTML
    )
    try:
        await context.bot.send_message(
            chat_id=target_user_id,
            text=f"🚫 You have been banned from using this bot.\nReason: {escape_html(reason)}", # Changed to HTML
            parse_mode=ParseMode.HTML # Changed to HTML
        )
    except Exception:
        pass
    db.log_action(user_id, 'ban_user', f"Banned {target_user_id} for: {reason}", status='success')
    return ConversationHandler.END

@admin_required
async def unban_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'

    if len(context.args) != 1:
        await update.message.reply_text(get_string(user_lang, "unban_command_usage"), parse_mode=ParseMode.HTML) # Changed to HTML
        return
    
    try:
        target_user_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text(get_string(user_lang, "invalid_user_id_format"))
        return
    
    user_profile = db.get_user(target_user_id)
    if not user_profile:
        await update.message.reply_text(get_string(user_lang, "user_not_found"))
        return

    if not user_profile.is_banned:
        await update.message.reply_text(f"❌ User {target_user_id} is not banned.")
        return

    user_profile.is_banned = False
    user_profile.ban_reason = ""
    db.save_user(user_profile)

    await update.message.reply_text(
        get_string(user_lang, "user_unbanned_success").format(user_id=target_user_id),
        parse_mode=ParseMode.HTML # Changed to HTML
    )
    try:
        await context.bot.send_message(
            chat_id=target_user_id,
            text="✅ You have been unbanned and can now use the bot again.",
            parse_mode=ParseMode.HTML # Changed to HTML
        )
    except Exception:
        pass
    db.log_action(user_id, 'unban_user', f"Unbanned {target_user_id}", status='success')

@admin_required
async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Broadcast command handler - initiates conversation for broadcast"""
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'

    return await broadcast_start(update, context)

@admin_required
async def system_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """System statistics command (admin only)"""
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    
    stats = db.get_statistics()
    
    uptime_seconds = int(time.time() - context.bot_data.get('start_time', time.time()))
    success_rate = (stats.successful_forwards / max(1, stats.total_forwards)) * 100 if stats.total_forwards > 0 else 0
    
    system_text = get_string(user_lang, "system_stats") + f"""

<b>📊 Forwarding:</b>
• Total Forwards: {stats.total_forwards:,}
• Successful: {stats.successful_forwards:,}
• Failed: {stats.failed_forwards:,}
• Success Rate: {success_rate:.2f}%

<b>👥 Users:</b>
• Total Users: {stats.total_users:,}
• Premium Users: {stats.premium_users}
• Admin Users: {stats.admin_users}
• Free Users: {stats.total_users - stats.premium_users - stats.admin_users}

<b>⚙️ Rules:</b>
• Total Rules: {stats.total_rules:,}
• Active Rules: {stats.active_rules}
• Inactive Rules: {stats.total_rules - stats.active_rules}

<b>🔧 System:</b>
• Uptime: {escape_html(format_duration(uptime_seconds))}
• Database Size: {os.path.getsize(DATABASE_URL) / 1024 / 1024:.2f} MB
• Log Files: {len(list(LOGS_DIR.glob('*.log')))}
• Version: 3.0.0-premium

<b>📈 Performance:</b>
• Memory Usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB
• CPU Usage: {psutil.Process().cpu_percent():.1f}%
• Status: 🟢 Healthy
"""
    
    keyboard = [
        [
            InlineKeyboardButton("🔄 Refresh", callback_data="refresh_system_stats"),
            InlineKeyboardButton("📊 Detailed View", callback_data="detailed_system_stats")
        ],
        [
            InlineKeyboardButton("📈 Performance", callback_data="performance_stats"),
            InlineKeyboardButton("🗂️ Logs", callback_data="system_logs")
        ],
        [
            InlineKeyboardButton("⚙️ Settings", callback_data="system_settings_menu"),
            InlineKeyboardButton("🔧 Maintenance", callback_data="maintenance_menu")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.message:
        await update.message.reply_text(
            system_text,
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=reply_markup
        )
    elif update.callback_query:
        await safe_edit_message(
            update.callback_query,
            system_text,
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=reply_markup
        )

@admin_required
async def bot_settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Settings management command (admin only)"""
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    
    settings = GLOBAL_SETTINGS
    
    settings_text = get_string(user_lang, "bot_settings") + f"""

<b>🔧 Core Settings:</b>
• Auto Forward: {'✅ Enabled' if settings['auto_forward'] else '❌ Disabled'}
• Phone Required: {'✅ Required' if settings['phone_required'] else '❌ Optional'}
• Premium Features: {'✅ Enabled' if settings['premium_features'] else '❌ Disabled'}
• Maintenance Mode: {'✅ Active' if settings.get('maintenance_mode') else '❌ Inactive'}

<b>🚦 Rate Limiting:</b>
• Rate Limit: {settings['rate_limit']} calls/minute
• Max Rules/User: {settings['max_rules_per_user']}

<b>🔗 Integration:</b>
• Webhook: {'✅ Enabled' if settings['webhook_enabled'] else '❌ Disabled'}
• Analytics: {'✅ Enabled' if settings['analytics_enabled'] else '❌ Disabled'}
• Notifications: {'✅ Enabled' if settings['notification_enabled'] else '❌ Disabled'}

<b>💾 Data & Backup:</b>
• Auto Backup: {'✅ Enabled' if settings['backup_enabled'] else '❌ Disabled'}
• Debug Mode: {'✅ Active' if settings.get('debug_mode') else '❌ Inactive'}
• Owner Console: {'✅ Enabled' if settings.get('owner_console_enabled') else '❌ Disabled'}
"""
    
    keyboard = [
        [
            InlineKeyboardButton("🔄 Toggle Auto Forward", callback_data="toggle_global_setting_auto_forward"),
            InlineKeyboardButton("📱 Toggle Phone Req.", callback_data="toggle_global_setting_phone_required")
        ],
        [
            InlineKeyboardButton("🚦 Set Rate Limit", callback_data="set_global_setting_rate_limit"),
            InlineKeyboardButton("💎 Toggle Premium Features", callback_data="toggle_global_setting_premium_features")
        ],
        [
            InlineKeyboardButton("🔗 Toggle Webhook", callback_data="toggle_global_setting_webhook_enabled"),
            InlineKeyboardButton("📊 Toggle Analytics", callback_data="toggle_global_setting_analytics_enabled")
        ],
        [
            InlineKeyboardButton("💾 Toggle Auto Backup", callback_data="toggle_global_setting_backup_enabled"),
            InlineKeyboardButton("🐛 Toggle Debug Mode", callback_data="toggle_global_setting_debug_mode")
        ],
        [
            InlineKeyboardButton("🔧 Maintenance Mode", callback_data="maintenance_menu"),
            InlineKeyboardButton("💻 Toggle Owner Console", callback_data="toggle_global_setting_owner_console_enabled")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.message:
        await update.message.reply_text(
            settings_text,
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=reply_markup
        )
    elif update.callback_query:
        await safe_edit_message(
            update.callback_query,
            settings_text,
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=reply_markup
        )

# ============================================================================
# OWNER COMMANDS
# ============================================================================

@owner_required
async def maintenance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maintenance mode command (owner only)"""
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    current_mode = GLOBAL_SETTINGS.get('maintenance_mode', False)
    
    maintenance_text = get_string(user_lang, "maintenance_mode_info") + f"""

<b>Current Status:</b> {'🟠 Active' if current_mode else '🟢 Inactive'}

<b>When maintenance mode is active:</b>
• Only owner can use the bot
• All forwarding rules are paused
• Users see maintenance message
• System updates can be performed safely
"""
    
    keyboard = [
        [
            InlineKeyboardButton(
                "🟠 Enable Maintenance" if not current_mode else "🟢 Disable Maintenance",
                callback_data="toggle_global_setting_maintenance_mode"
            )
        ],
        [
            InlineKeyboardButton("📋 Maintenance Log", callback_data="maintenance_log"),
            InlineKeyboardButton("⏰ Schedule Maintenance", callback_data="schedule_maintenance")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.message:
        await update.message.reply_text(
            maintenance_text,
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=reply_markup
        )
    elif update.callback_query:
        await safe_edit_message(
            update.callback_query,
            maintenance_text,
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=reply_markup
        )

@owner_required
async def backup_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Backup command (owner only)"""
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    
    backup_text = get_string(user_lang, "system_backup_info") + """

<b>Backup Components:</b>
• User database
• Forwarding rules
• System settings
• Statistics & logs
• Configuration files

<b>Backup Schedule:</b>
• Automatic: Daily at 02:00 UTC (if enabled)
• Manual: On-demand
• Retention: 30 days
"""
    
    keyboard = [
        [
            InlineKeyboardButton("💾 Create Backup", callback_data="create_backup"),
            InlineKeyboardButton("📥 Download Latest", callback_data="download_latest_backup")
        ],
        [
            InlineKeyboardButton("🔄 Restore Backup", callback_data="restore_backup_menu"),
            InlineKeyboardButton("⚙️ Backup Settings", callback_data="backup_settings_menu")
        ],
        [
            InlineKeyboardButton("📊 Backup Stats", callback_data="backup_stats"),
            InlineKeyboardButton("🗂️ Manage Backups", callback_data="manage_backups")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.message:
        await update.message.reply_text(
            backup_text,
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=reply_markup
        )
    elif update.callback_query:
        await safe_edit_message(
            update.callback_query,
            backup_text,
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=reply_markup
        )

@owner_required
async def restore_backup_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'

    backup_files = sorted(BACKUP_DIR.glob("db_backup_*.db"), reverse=True)
    
    message_to_edit = update.effective_message # Use effective_message

    if not backup_files:
        if update.callback_query:
            await safe_edit_message(update.callback_query, get_string(user_lang, "no_backups_found"), parse_mode=ParseMode.HTML) # Changed to HTML
        else:
            await message_to_edit.reply_text(get_string(user_lang, "no_backups_found"), parse_mode=ParseMode.HTML) # Changed to HTML
        return

    text = get_string(user_lang, "available_backups") + "\n\nSelect a backup to restore:\n"
    keyboard = []
    for f in backup_files[:10]: # Show latest 10
        keyboard.append([InlineKeyboardButton(f.name, callback_data=f"restore_confirm_{f.name}")])
    
    keyboard.append([InlineKeyboardButton("🔙 Back to Backup Menu", callback_data="backup_menu")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.callback_query:
        await safe_edit_message(update.callback_query, text, reply_markup=reply_markup, parse_mode=ParseMode.HTML) # Changed to HTML
    else:
        await message_to_edit.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML) # Changed to HTML


@owner_required
async def restore_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    
    backup_file_name = update.callback_query.data.replace("restore_confirm_", "")
    backup_path = BACKUP_DIR / backup_file_name

    if not backup_path.exists():
        await safe_edit_message(update.callback_query, f"❌ Backup file not found: {escape_html(backup_file_name)}", parse_mode=ParseMode.HTML) # Changed to HTML
        return

    keyboard = [
        [InlineKeyboardButton("✅ Confirm Restore", callback_data=f"restore_execute_{backup_file_name}")],
        [InlineKeyboardButton("❌ Cancel", callback_data="restore_backup_menu")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await safe_edit_message(
        update.callback_query,
        f"⚠️ <b>WARNING: This will overwrite the current bot data!</b> Are you sure you want to restore from <code>{escape_html(backup_file_name)}</code>?", # Changed to HTML
        parse_mode=ParseMode.HTML, # Changed to HTML
        reply_markup=reply_markup
    )

@owner_required
async def restore_execute(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global db  # Declare global at the very beginning
    
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    
    backup_file_name = update.callback_query.data.replace("restore_execute_", "")
    backup_path = BACKUP_DIR / backup_file_name

    try:
        # Stop the bot's polling/webhook before restoring
        await context.application.stop()
        
        # Close existing DB connection
        del db # Remove reference to old DBManager
        
        # Perform the restore
        shutil.copyfile(backup_path, DATABASE_URL)
        
        # Reinitialize DB connection
        db = DatabaseManager(DATABASE_URL)
        forwarding_engine.reload_rules() # Reload rules from new DB
        
        await safe_edit_message(
            update.callback_query,
            get_string(user_lang, "restore_success").format(backup_file_name=escape_html(backup_file_name)), # Changed to HTML
            parse_mode=ParseMode.HTML # Changed to HTML
        )
        db.log_action(user_id, 'restore_backup', f"Restored from {backup_file_name}", status='success')
        
        # Restart the bot (this will be handled by the main loop if it's designed to restart on stop)
        # For a simple script, you might need to exit and rely on a process manager to restart.
        # For now, we'll just log and let the main loop handle the restart.
        logger.info("Bot stopped for restore. Please restart the bot process manually if it doesn't auto-restart.")
        # In a production setup, you'd typically have a process manager (systemd, supervisor) that restarts the script.
        # For this example, we'll just exit.
        os._exit(0)

    except Exception as e:
        logger.error(f"Restore failed: {e}")
        await safe_edit_message(
            update.callback_query,
            get_string(user_lang, "restore_failed").format(error=escape_html(str(e))), # Changed to HTML
            parse_mode=ParseMode.HTML # Changed to HTML
        )
        db.log_action(user_id, 'restore_backup', f"Failed to restore from {backup_file_name}", status='error', error_message=str(e))
        # Re-initialize DB and engine in case of failure to ensure bot is functional
        db = DatabaseManager(DATABASE_URL)
        forwarding_engine.reload_rules()



@owner_required
async def debug_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Debug information command (owner only)"""
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    
    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
    cpu_percent = psutil.Process().cpu_percent()
    
    debug_text = get_string(user_lang, "debug_info") + f"""

<b>🐍 Python Environment:</b>
• Python Version: {escape_html(sys.version.split()[0])}
• Telegram Bot API: {escape_html(TG_VER)}
• Platform: {escape_html(sys.platform)}

<b>💾 Memory & Performance:</b>
• Memory Usage: {memory_usage:.1f} MB
• CPU Usage: {cpu_percent:.1f}%
• Active Rules: {len(forwarding_engine.active_rules)}
• In-memory Forwards: {forwarding_engine.stats_in_memory['total_forwards']}

<b>🗃️ Database:</b>
• File Size: {os.path.getsize(DATABASE_URL) / 1024:.1f} KB
• Connection Status: ✅ Connected

<b>🔧 Configuration:</b>
• Bot Token: {'✅ Valid' if BOT_TOKEN else '❌ Missing'}
• Owner ID: {OWNER_ID}
• Webhook: {'✅ Configured' if GLOBAL_SETTINGS.get('webhook_enabled') else '❌ Not set'}
• Log Level: {escape_html(LOG_LEVEL)}
"""
    
    keyboard = [
        [
            InlineKeyboardButton("🔍 Detailed Debug", callback_data="detailed_debug"),
            InlineKeyboardButton("📋 Export Debug", callback_data="export_debug")
        ],
        [
            InlineKeyboardButton("🧪 Test Features", callback_data="test_features"),
            InlineKeyboardButton("🔄 Reload Config", callback_data="reload_config")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.message:
        await update.message.reply_text(
            debug_text,
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=reply_markup
        )
    elif update.callback_query:
        await safe_edit_message(
            update.callback_query,
            debug_text,
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=reply_markup
        )

@owner_required
async def clear_logs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'

    try:
        cutoff_date = datetime.now() - timedelta(days=30)
        with sqlite3.connect(DATABASE_URL) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM logs WHERE timestamp < ?", (cutoff_date.isoformat(),))
            conn.commit()
        
        text = get_string(user_lang, "clear_logs_success")
        db.log_action(user_id, 'clear_logs', 'Cleared logs older than 30 days', status='success')
    except Exception as e:
        text = get_string(user_lang, "clear_logs_failed").format(error=escape_html(str(e))) # Changed to HTML
        db.log_action(user_id, 'clear_logs', 'Failed to clear logs', status='error', error_message=str(e))
    
    if update.message:
        await update.message.reply_text(text, parse_mode=ParseMode.HTML) # Changed to HTML
    elif update.callback_query:
        await safe_edit_message(update.callback_query, text, parse_mode=ParseMode.HTML) # Changed to HTML

# ============================================================================
# MESSAGE HANDLERS
# ============================================================================

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all non-command messages for auto-forwarding"""
    
    # Check if this is a channel post (channels don't have users)
    if update.channel_post:
        # Channel posts don't have users, but we can still forward them
        if GLOBAL_SETTINGS.get('auto_forward', True):
            await forwarding_engine.process_message(update, context)
        return
    
    # For regular messages, check if we have both message and user
    if not update.effective_message:
        logger.debug("No effective message in update")
        return
        
    if not update.effective_user:
        logger.debug("No effective user in update (might be a channel)")
        return
    
    user_id = update.effective_user.id
    user_profile = db.get_user(user_id)
    user_lang = user_profile.settings.get('language', 'en') if user_profile else 'en'

    if user_profile and user_profile.is_banned:
        return

    if GLOBAL_SETTINGS.get('maintenance_mode', False):
        if user_id != OWNER_ID:
            await update.effective_message.reply_text(get_string(user_lang, "bot_under_maintenance"))
            return
    
    if GLOBAL_SETTINGS.get('auto_forward', True):
        await forwarding_engine.process_message(update, context)

# ============================================================================
# CALLBACK QUERY HANDLERS
# ============================================================================

# If it's in the callback_query_handler function:
async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global GLOBAL_SETTINGS  # Declare global at the very beginning
    
    query = update.callback_query
    await query.answer()
    
    data = query.data
    user_id = query.from_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'

    # Now you can safely use GLOBAL_SETTINGS
    if data == "reload_config":
        GLOBAL_SETTINGS = db.get_all_global_settings()
        forwarding_engine.reload_rules()
        await safe_edit_message(query, "🔄 Configuration reloaded from database and rules reloaded.", # Changed to HTML
                                      reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Debug Menu", callback_data="debug_menu")]])
        )
   

    # --- General Navigation ---
    elif data == "help":
        await help_command(update, context)
    elif data == "quickstart":
        quickstart_text = """
🚀 <b>Quick Start Guide</b>

<b>Step 1: Phone Verification</b>
Use /connect_phone to verify your phone number (required for premium features).

<b>Step 2: Create Your First Rule</b>
Use /autoforward and follow the interactive steps.

<b>Step 3: Test Manual Forward</b>
Reply to any message and use <code>/forward without_tags -1009876543210</code>

<b>Step 4: Manage Rules</b>
Use /rules to view and manage your forwarding rules.

<b>🎯 Forwarding Modes:</b>
• <code>with_tags</code> - Keeps "Forwarded from" info
• <code>without_tags</code> - Clean copy without forward tags
• <code>bypass</code> - Bypass restrictions + clean copy

<b>💡 Pro Tips:</b>
• Use negative IDs for channels/groups
• Start with without_tags mode
• Monitor your rules with /stats
• Use keywords for filtered forwarding
"""
        keyboard = [
            [
                InlineKeyboardButton("📱 Connect Phone", callback_data="connect_phone"),
                InlineKeyboardButton("📝 My Rules", callback_data="my_rules")
            ],
            [
                InlineKeyboardButton("💎 Premium Features", callback_data="premium_info"),
                InlineKeyboardButton("🆘 Get Help", url="https://t.me/YourSupportBot")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await safe_edit_message(query, quickstart_text, parse_mode=ParseMode.HTML, reply_markup=reply_markup) # Changed to HTML

    
    elif data == "premium_info":
        premium_text = """
💎 <b>Premium Features</b>

<b>🚀 Advanced Forwarding:</b>
• Unlimited auto-forwarding rules
• Keyword filtering
• Text replacement
• Scheduled forwarding
• Batch operations

<b>📊 Analytics & Stats:</b>
• Detailed forwarding statistics
• Success rate monitoring
• Activity logs
• Export capabilities

<b>⚙️ Customization:</b>
• Custom message templates
• Advanced filters
• Rule priorities
• Conditional forwarding

<b>🛡️ Security & Control:</b>
• Phone verification
• Access control
• Rate limiting bypass
• Priority support

<b>🎯 Special Modes:</b>
• Bypass restrictions
• Silent forwarding
• Media-only forwarding
• Text-only forwarding

<b>📱 Contact admin to upgrade your account!</b>
"""
        keyboard = [
            [
                InlineKeyboardButton("🎖️ Request Premium", url="https://t.me/YourAdminBot"),
                InlineKeyboardButton("📊 View Pricing", callback_data="pricing_info")
            ],
            [
                InlineKeyboardButton("🔙 Back to Help", callback_data="help")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await safe_edit_message(query, premium_text, parse_mode=ParseMode.HTML, reply_markup=reply_markup) # Changed to HTML
    
    elif data == "connect_phone":
        await connect_phone_command(update, context)
    elif data == "my_rules":
        await rules_command(update, context)
    elif data == "user_stats":
        await stats_command(update, context)
    elif data == "admin_panel":
        await users_command(update, context) # Redirect to users management as admin panel entry
    elif data == "refresh_status":
        await status_command(update, context)
    elif data == "start":
        await start_command(update, context)
    elif data == "refresh_logs":
        await logs_command(update, context)
    elif data == "user_settings":
        await user_settings_command(update, context)

    # --- User Settings Callbacks ---
    elif data.startswith("toggle_user_setting_"):
        setting_key = data.replace("toggle_user_setting_", "")
        user_profile = db.get_user(user_id)
        if user_profile:
            current_value = user_profile.settings.get(setting_key, True if setting_key == "notifications" else False)
            user_profile.settings[setting_key] = not current_value
            db.save_user(user_profile)
            await safe_edit_message(
                query,
                f"✅ Setting '<code>{escape_html(setting_key)}</code>' updated to '<code>{'Enabled' if not current_value else 'Disabled'}</code>'.", # Changed to HTML
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Settings", callback_data="user_settings")]])
            )
        else:
            await safe_edit_message(query, "❌ User profile not found.")
    
    elif data.startswith("change_user_setting_"):
        setting_key = data.replace("change_user_setting_", "")
        context.user_data['setting_to_change'] = setting_key
        
        if setting_key == "language":
            keyboard = [
                [InlineKeyboardButton("English", callback_data="set_user_setting_language_en")],
                [InlineKeyboardButton("Español", callback_data="set_user_setting_language_es")],
                [InlineKeyboardButton("Français", callback_data="set_user_setting_language_fr")],
                [InlineKeyboardButton("Deutsch", callback_data="set_user_setting_language_de")],
                [InlineKeyboardButton("Русский", callback_data="set_user_setting_language_ru")],
                [InlineKeyboardButton("Italiano", callback_data="set_user_setting_language_it")],
                [InlineKeyboardButton("🔙 Back", callback_data="user_settings")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await safe_edit_message(query, "Select your language:", reply_markup=reply_markup)
        elif setting_key == "timezone":
            await safe_edit_message(query, "Please send your timezone (e.g., <code>America/New_York</code> or <code>UTC+2</code>). Use <code>/cancel</code> to abort.", parse_mode=ParseMode.HTML) # Changed to HTML
            context.user_data['current_state'] = SETTING_CHANGE_VALUE # Set state for text input
            return SETTING_CHANGE_VALUE
        elif setting_key == "privacy_mode":
            keyboard = [
                [InlineKeyboardButton("Default (Show all info)", callback_data="set_user_setting_privacy_mode_default")],
                [InlineKeyboardButton("Strict (Hide sensitive info)", callback_data="set_user_setting_privacy_mode_strict")],
                [InlineKeyboardButton("🔙 Back", callback_data="user_settings")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await safe_edit_message(query, "Select privacy mode:", reply_markup=reply_markup)
        else:
            await safe_edit_message(query, "❌ This setting cannot be changed via this menu.")
        
    elif data.startswith("set_user_setting_"):
        parts = data.split('_')
        setting_key = parts[3]
        value = parts[4]
        
        user_profile = db.get_user(user_id)
        if user_profile:
            user_profile.settings[setting_key] = value
            db.save_user(user_profile)
            await safe_edit_message(
                query,
                f"✅ Setting '<code>{escape_html(setting_key)}</code>' updated to '<code>{escape_html(value)}</code>'.", # Changed to HTML
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Settings", callback_data="user_settings")]])
            )
        else:
            await safe_edit_message(query, "❌ User profile not found.")

    # --- Rule Management Callbacks ---
    elif data == "new_rule":
        # Start the conversation for rule creation
        await safe_edit_message(query, "Starting new rule creation...")
        # Directly call the entry point of the conversation handler
        return await create_rule_start(update, context)
    elif data == "refresh_rules":
        await rules_command(update, context)
    elif data == "manage_all_rules":
        await manage_all_rules_command(update, context)
    elif data.startswith("manage_all_rules_page_"):
        page = int(data.split('_')[-1])
        await manage_all_rules_command(update, context, page)
    elif data.startswith("rule_settings_"):
        rule_id = data.replace("rule_settings_", "")
        await edit_rule_start(update, context, rule_id)
        return
    elif data.startswith("edit_rule_"): # All other edit rule callbacks
        await edit_rule_callback_handler(update, context)
        return
    elif data.startswith("confirm_delete_rule_"):
        await edit_rule_callback_handler(update, context)
        return
    elif data.startswith("set_mode_"):
        await edit_rule_callback_handler(update, context)
        return
    elif data.startswith("edit_rule_filter_"):
        await edit_rule_callback_handler(update, context)
        return
    elif data.startswith("edit_rule_schedule_"):
        await edit_rule_callback_handler(update, context)
        return

    # --- Statistics Callbacks ---
    elif data == "detailed_user_stats":
        stats = db.get_statistics()
        user_profile = db.get_user(user_id)
        
        text = f"📊 <b>Detailed User Statistics for {escape_html(user_profile.first_name)}</b>\n\n" # Changed to HTML
        text += "<b>Daily Forwards (Last 7 Days):</b>\n" # Changed to HTML
        today = datetime.now().date()
        for i in range(7):
            date = today - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            daily_data = stats.daily_stats.get(date_str, {"forwards": 0, "successful": 0, "failed": 0})
            text += f"• {escape_html(date_str)}: Total {daily_data['forwards']}, Success {daily_data['successful']}, Failed {daily_data['failed']}\n" # Changed to HTML
        
        keyboard = [[InlineKeyboardButton("🔙 Back to My Stats", callback_data="user_stats")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await safe_edit_message(query, text, parse_mode=ParseMode.HTML, reply_markup=reply_markup) # Changed to HTML

    elif data == "stats_charts":
        stats = db.get_statistics()
        text = "📈 <b>Statistics Charts (Text-based)</b>\n\n" # Changed to HTML
        text += "<b>Daily Forwards (Last 7 Days Bar Chart):</b>\n" # Changed to HTML
        
        daily_data_points = []
        today = datetime.now().date()
        for i in range(7):
            date = today - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            daily_data_points.append(stats.daily_stats.get(date_str, {"forwards": 0}).get("forwards", 0))
        
        max_val = max(daily_data_points) if daily_data_points else 1
        scale = 20 / max_val if max_val > 0 else 0 # Scale to 20 characters
        
        for i in range(7):
            date = today - timedelta(days=i)
            date_str = date.strftime('%m-%d')
            value = daily_data_points[i]
            bar = "█" * int(value * scale)
            text += f"{escape_html(date_str)}: {escape_html(bar)} {value}\n" # Changed to HTML
        
        keyboard = [[InlineKeyboardButton("🔙 Back to My Stats", callback_data="user_stats")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await safe_edit_message(query, text, parse_mode=ParseMode.HTML, reply_markup=reply_markup) # Changed to HTML

    elif data == "export_user_data":
        user_profile = db.get_user(user_id)
        user_rules = db.get_rules(user_id)
        user_logs = db.get_logs(user_id=user_id, limit=500) # Export more logs

        export_data = {
            "user_profile": asdict(user_profile) if user_profile else {},
            "rules": [asdict(rule) for rule in user_rules],
            "logs": user_logs
        }
        
        export_filename = f"user_data_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_path = TEMP_DIR / export_filename
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            await context.bot.send_document(
                chat_id=user_id,
                document=InputFile(str(export_path)),
                caption="✅ Your exported data is attached."
            )
            os.remove(export_path) # Clean up temp file
        except Exception as e:
            await safe_edit_message(query, f"❌ Failed to export data: {escape_html(str(e))}", parse_mode=ParseMode.HTML) # Changed to HTML
            logger.error(f"Failed to export user data for {user_id}: {e}")
        
        await safe_edit_message(query, "✅ Data export initiated. Check your chat for the file.", # Changed to HTML
                                      reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to My Stats", callback_data="user_stats")]])
        )

    elif data == "refresh_user_stats":
        await stats_command(update, context)

    # --- Admin Callbacks ---
    elif data == "user_search":
        await safe_edit_message(query, "👤 <b>User Search</b>\n\nPlease send the User ID, Username, or First Name to search for. Use <code>/cancel</code> to abort.", parse_mode=ParseMode.HTML) # Changed to HTML
        context.user_data['current_state'] = "user_search_input"
        return SETTING_CHANGE_VALUE # Re-use state for text input
    
    elif data == "admin_user_stats":
        stats = db.get_statistics()
        text = f"📊 <b>Admin User Statistics Overview</b>\n\n" \
               f"• Total Users: {stats.total_users:,}\n" \
               f"• Premium Users: {stats.premium_users}\n" \
               f"• Admin Users: {stats.admin_users}\n" \
               f"• Banned Users: {len([u for u in db.get_all_users() if u.is_banned])}\n\n" \
               f"<b>New Users by Day (Last 7 Days):</b>\n" # Changed to HTML
        
        today = datetime.now().date()
        for i in range(7):
            date = today - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            daily_data = stats.daily_stats.get(date_str, {"users_joined": 0})
            text += f"• {escape_html(date_str)}: {daily_data.get('users_joined', 0)} new users\n" # Changed to HTML

        keyboard = [[InlineKeyboardButton("🔙 Back to Users Menu", callback_data="admin_panel")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await safe_edit_message(query, text, parse_mode=ParseMode.HTML, reply_markup=reply_markup) # Changed to HTML

    elif data == "promote_user_menu":
        await safe_edit_message(
            query,
            get_string(user_lang, "promote_command_usage"),
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Users Menu", callback_data="admin_panel")]])
        )
    elif data == "demote_user_menu":
        await safe_edit_message(
            query,
            "📝 <b>Demote User Usage:</b>\n\n" # Changed to HTML
            "<code>/demote &lt;user_id&gt; [role]</code>\n\n" # Changed to HTML
            "Default role is <code>free</code>. Available roles: <code>free</code>, <code>premium</code>\n\n" # Changed to HTML
            "Example: <code>/demote 123456789 free</code>", # Changed to HTML
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Users Menu", callback_data="admin_panel")]])
        )
    elif data == "ban_user_menu":
        await safe_edit_message(
            query,
            get_string(user_lang, "ban_command_usage"),
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Users Menu", callback_data="admin_panel")]])
        )
    elif data == "unban_user_menu":
        await safe_edit_message(
            query,
            get_string(user_lang, "unban_command_usage"),
            parse_mode=ParseMode.HTML, # Changed to HTML
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Users Menu", callback_data="admin_panel")]])
        )
    elif data == "broadcast_menu":
        # Call broadcast_start, which handles both command and callback entry
        return await broadcast_start(update, context)
    elif data.startswith("broadcast_"): # For broadcast target selection
        await broadcast_select_target(update, context)
        return # Conversation continues

    elif data == "refresh_system_stats":
        await system_stats_command(update, context)
    elif data == "detailed_system_stats":
        stats = db.get_statistics()
        text = f"📊 <b>Detailed System Statistics</b>\n\n" \
               f"<b>Daily Forwards (Last 7 Days):</b>\n" # Changed to HTML
        today = datetime.now().date()
        for i in range(7):
            date = today - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            daily_data = stats.daily_stats.get(date_str, {"total": 0, "successful": 0, "failed": 0})
            text += f"• {escape_html(date_str)}: Total {daily_data.get('forwards',0)}, Success {daily_data.get('successful',0)}, Failed {daily_data.get('failed',0)}\n" # Changed to HTML
        
        keyboard = [[InlineKeyboardButton("🔙 Back to System Stats", callback_data="system_stats")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await safe_edit_message(query, text, parse_mode=ParseMode.HTML, reply_markup=reply_markup) # Changed to HTML

    elif data == "performance_stats":
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_percent = psutil.Process().cpu_percent()
        text = f"📈 <b>System Performance Metrics</b>\n\n" \
               f"• Memory Usage: {memory_usage:.1f} MB\n" \
               f"• CPU Usage: {cpu_percent:.1f}%\n" \
               f"• Active Threads: {threading.active_count()}\n" \
               f"• Active Rules in Memory: {len(forwarding_engine.active_rules)}\n" \
               f"• In-memory Forward Attempts (since last restart): {forwarding_engine.stats_in_memory['total_forwards']}"
        keyboard = [[InlineKeyboardButton("🔙 Back to System Stats", callback_data="system_stats")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await safe_edit_message(query, text, parse_mode=ParseMode.HTML, reply_markup=reply_markup) # Changed to HTML

    elif data == "system_logs":
        logs = db.get_logs(limit=20) # Get latest 20 system logs
        if not logs:
            text = "📋 <b>System Activity Logs</b>\n\nNo recent system activity found." # Changed to HTML
        else:
            text = "📋 <b>System Activity Logs</b>\n\n" # Changed to HTML
            for log in logs:
                timestamp = datetime.fromisoformat(log['timestamp']).strftime('%Y-%m-%d %H:%M')
                status_emoji = "✅" if log['status'] == 'success' else "❌"
                text += f"• {escape_html(timestamp)} {status_emoji} <b>{escape_html(log['action'].replace('_', ' ').title())}</b>: {escape_html(log['details'])}\n" # Changed to HTML
                if log['error_message']:
                    text += f"  <i>Error: {escape_html(truncate_text(log['error_message'], 50))}</i>\n" # Changed to HTML
        keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="system_logs")],
                    [InlineKeyboardButton("🔙 Back to System Stats", callback_data="system_stats")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await safe_edit_message(query, text, parse_mode=ParseMode.HTML, reply_markup=reply_markup) # Changed to HTML

    elif data == "system_settings_menu":
        await bot_settings_command(update, context)

    # --- Global Settings Callbacks ---
    elif data.startswith("toggle_global_setting_"):
        setting_key = data.replace("toggle_global_setting_", "")
        current_value = GLOBAL_SETTINGS.get(setting_key)
        GLOBAL_SETTINGS[setting_key] = not current_value
        db.set_global_setting(setting_key, GLOBAL_SETTINGS[setting_key])
        
        if setting_key == "maintenance_mode":
            forwarding_engine.reload_rules() # Rules are paused in maintenance mode
            status_msg = f"{'⚠️ Bot is now in maintenance mode. Only owner can use it.' if GLOBAL_SETTINGS[setting_key] else '✅ Bot is back online for all users.'}"
        else:
            status_msg = ""

        await safe_edit_message(
            query,
            get_string(user_lang, "settings_updated").format(key=escape_html(setting_key), value=escape_html(str(GLOBAL_SETTINGS[setting_key]))) + f"\n{status_msg}", # Changed to HTML
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Settings", callback_data="system_settings_menu")]])
        )
    
    elif data.startswith("set_global_setting_"):
        setting_key = data.replace("set_global_setting_", "")
        context.user_data['setting_to_change'] = setting_key
        
        if setting_key == "rate_limit":
            await safe_edit_message(query, f"Please send the new rate limit (e.g., <code>60</code> for 60 calls/minute). Current: {GLOBAL_SETTINGS.get('rate_limit')}. Use <code>/cancel</code> to abort.", parse_mode=ParseMode.HTML) # Changed to HTML
            context.user_data['current_state'] = SETTING_CHANGE_VALUE
            return SETTING_CHANGE_VALUE
        else:
            await safe_edit_message(query, "❌ This setting cannot be changed via this menu.")

    # --- Owner Callbacks ---
    elif data == "maintenance_menu":
        await maintenance_command(update, context)
    elif data == "backup_menu":
        await backup_command(update, context)
    elif data == "create_backup":
        console = OwnerConsole(context.application)
        await safe_edit_message(query, "⏳ Creating backup...")
        try:
            backup_name, db_backup_name = console._create_backup_sync() # Call sync version for callback
            await safe_edit_message(
                query,
                get_string(user_lang, "backup_created").format(backup_name=escape_html(backup_name), db_backup_name=escape_html(db_backup_name)), # Changed to HTML
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Backup Menu", callback_data="backup_menu")]])
            )
        except Exception as e:
            await safe_edit_message(
                query,
                get_string(user_lang, "backup_failed").format(error=escape_html(str(e))), # Changed to HTML
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Backup Menu", callback_data="backup_menu")]])
            )
    elif data == "download_latest_backup":
        backup_files = sorted(BACKUP_DIR.glob("db_backup_*.db"), reverse=True)
        if backup_files:
            latest_backup = backup_files[0]
            try:
                await context.bot.send_document(
                    chat_id=user_id,
                    document=InputFile(str(latest_backup)),
                    caption=f"✅ Latest database backup: <code>{escape_html(latest_backup.name)}</code>" # Changed to HTML
                )
                await safe_edit_message(
                    query,
                    f"✅ Latest backup <code>{escape_html(latest_backup.name)}</code> sent to your chat.", # Changed to HTML
                    parse_mode=ParseMode.HTML, # Changed to HTML
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Backup Menu", callback_data="backup_menu")]])
                )
            except Exception as e:
                await safe_edit_message(query, f"❌ Failed to send backup: {escape_html(str(e))}", parse_mode=ParseMode.HTML) # Changed to HTML
        else:
            await safe_edit_message(query, get_string(user_lang, "no_backups_found"))
    elif data == "restore_backup_menu":
        await restore_backup_menu(update, context)
    elif data.startswith("restore_confirm_"):
        await restore_confirm(update, context)
    elif data.startswith("restore_execute_"):
        await restore_execute(update, context)
    elif data == "backup_settings_menu":
        await safe_edit_message(query, "⚙️ Backup settings (e.g., retention, frequency) coming soon!",
                                      reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Backup Menu", callback_data="backup_menu")]])
        )
    elif data == "backup_stats":
        await safe_edit_message(query, "📊 Backup statistics coming soon!",
                                      reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Backup Menu", callback_data="backup_menu")]])
        )
    elif data == "manage_backups":
        await safe_edit_message(query, "🗂️ Manage backups (list, delete) coming soon!",
                                      reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Backup Menu", callback_data="backup_menu")]])
        )
    elif data == "detailed_debug":
        await safe_edit_message(query, "🔍 Detailed debug info (e.g., full logs, DB schema) coming soon!",
                                      reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Debug Menu", callback_data="debug_menu")]])
        )
    elif data == "export_debug":
        await safe_edit_message(query, "📋 Export debug info (e.g., full logs, system info) coming soon!",
                                      reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Debug Menu", callback_data="debug_menu")]])
        )
    elif data == "test_features":
        await safe_edit_message(query, "🧪 Test features (e.g., send test message, trigger rule) coming soon!",
                                      reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Debug Menu", callback_data="debug_menu")]])
        )
    elif data == "reload_config":
        # Reload global settings from DB
          # This should be at the beginning of the handler
        GLOBAL_SETTINGS = db.get_all_global_settings()
        forwarding_engine.reload_rules()
        await safe_edit_message(query, "🔄 Configuration reloaded from database and rules reloaded.", # Changed to HTML
                                  reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Debug Menu", callback_data="debug_menu")]])
    )

    elif data == "debug_menu": # For back button
        await debug_command(update, context)
    elif data == "maintenance_log":
        await safe_edit_message(query, "📋 Maintenance log coming soon!",
                                      reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Maintenance Menu", callback_data="maintenance_menu")]])
        )
    elif data == "schedule_maintenance":
        await safe_edit_message(query, "⏰ Schedule maintenance coming soon!",
                                      reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Maintenance Menu", callback_data="maintenance_menu")]])
        )
    elif data == "phone_setup":
        await safe_edit_message(query, "📱 Phone setup details coming soon!",
                                      reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Help", callback_data="help")]])
        )
    elif data == "examples":
        await safe_edit_message(query, "⚡ Examples coming soon!",
                                      reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Help", callback_data="help")]])
        )
    elif data == "pricing_info":
        await safe_edit_message(query, "📊 Pricing information coming soon!",
                                      reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Premium Info", callback_data="premium_info")]])
        )
    else:
        await safe_edit_message(
            query,
            "⚠️ This feature is not yet implemented or is under development.",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔙 Back", callback_data="help")
            ]])
        )

async def handle_setting_change_value(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'
    
    setting_key = context.user_data.get('setting_to_change')
    user_input = update.message.text.strip()

    if setting_key == "timezone":
        user_profile = db.get_user(user_id)
        if user_profile:
            user_profile.settings['timezone'] = user_input
            db.save_user(user_profile)
            await update.message.reply_text(
                f"✅ Your timezone has been set to <code>{escape_html(user_input)}</code>.", # Changed to HTML
                parse_mode=ParseMode.HTML, # Changed to HTML
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Settings", callback_data="user_settings")]])
            )
        else:
            await update.message.reply_text("❌ User profile not found.")
    elif setting_key == "rate_limit":
        try:
            new_rate_limit = int(user_input)
            if new_rate_limit <= 0:
                raise ValueError("Rate limit must be positive.")
            GLOBAL_SETTINGS['rate_limit'] = new_rate_limit
            db.set_global_setting('rate_limit', new_rate_limit)
            await update.message.reply_text(
                get_string(user_lang, "settings_updated").format(key=escape_html('rate_limit'), value=escape_html(str(new_rate_limit))), # Changed to HTML
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Settings", callback_data="system_settings_menu")]])
            )
            # Update the application's rate limiter
            context.application.rate_limiter.overall_max_rate = new_rate_limit
        except ValueError:
            await update.message.reply_text(get_string(user_lang, "invalid_setting_value").format(key=escape_html('rate_limit'))) # Changed to HTML
            return SETTING_CHANGE_VALUE # Stay in this state
    elif context.user_data.get('current_state') == "user_search_input":
        search_query = user_input.lower()
        all_users = db.get_all_users()
        found_users = []
        
        for user_profile in all_users:
            if str(user_profile.user_id) == search_query or \
               (user_profile.username and search_query in user_profile.username.lower()) or \
               (user_profile.first_name and search_query in user_profile.first_name.lower()) or \
               (user_profile.last_name and search_query in user_profile.last_name.lower()):
                found_users.append(user_profile)
        
        if not found_users:
            await update.message.reply_text("❌ No users found matching your query.")
        else:
            text = "👤 <b>Search Results:</b>\n\n" # Changed to HTML
            keyboard = []
            for user_profile in found_users[:10]: # Show up to 10 results
                text += f"• ID: <code>{user_profile.user_id}</code>\n" \
                        f"  Name: {escape_html(user_profile.first_name)} {escape_html(user_profile.last_name or '')} (@{escape_html(user_profile.username or 'N/A')})\n" \
                        f"  Role: {escape_html(user_profile.role.title())}\n" \
                        f"  Banned: {'✅' if user_profile.is_banned else '❌'}\n\n"
                keyboard.append([InlineKeyboardButton(f"View {user_profile.first_name} ({user_profile.user_id})", callback_data=f"view_user_profile_{user_profile.user_id}")])
            
            keyboard.append([InlineKeyboardButton("🔙 Back to Users Menu", callback_data="admin_panel")])
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=reply_markup) # Changed to HTML
    else:
        await update.message.reply_text(get_string(user_lang, "invalid_input"))
        return ConversationHandler.END # End if unknown state

    context.user_data.clear()
    return ConversationHandler.END

async def view_user_profile_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    user_lang = db.get_user(user_id).settings.get('language', 'en') if db.get_user(user_id) else 'en'

    target_user_id = int(query.data.replace("view_user_profile_", ""))
    user_profile = db.get_user(target_user_id)

    if not user_profile:
        await safe_edit_message(query, get_string(user_lang, "user_not_found"))
        return

    text = get_string(user_lang, "user_info").format(user_name=escape_html(user_profile.first_name)) + f"""
• ID: <code>{user_profile.user_id}</code>
• Username: @{escape_html(user_profile.username or 'N/A')}
• Full Name: {escape_html(user_profile.first_name)} {escape_html(user_profile.last_name or '')}
• Role: {escape_html(user_profile.role.title())}
• Phone: {escape_html(user_profile.phone or 'N/A')}
• Joined: {escape_html(datetime.fromisoformat(user_profile.joined_date).strftime('%Y-%m-%d %H:%M'))}
• Last Active: {escape_html(datetime.fromisoformat(user_profile.last_active).strftime('%Y-%m-%d %H:%M'))}
• Total Forwards: {user_profile.total_forwards:,}
• Total Rules: {user_profile.total_rules}
• Banned: {'✅' if user_profile.is_banned else '❌'}
• Ban Reason: {escape_html(user_profile.ban_reason or 'N/A')}
"""
    keyboard = [
        [InlineKeyboardButton("🎖️ Promote", callback_data=f"promote_user_id_{target_user_id}"),
         InlineKeyboardButton("📉 Demote", callback_data=f"demote_user_id_{target_user_id}")],
        [InlineKeyboardButton("🚫 Ban", callback_data=f"ban_user_id_{target_user_id}"),
         InlineKeyboardButton("✅ Unban", callback_data=f"unban_user_id_{target_user_id}")],
        [InlineKeyboardButton("🔙 Back to Search", callback_data="user_search")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await safe_edit_message(query, text, parse_mode=ParseMode.HTML, reply_markup=reply_markup) # Changed to HTML

# ============================================================================
# OWNER CONSOLE (STDIN)
# ============================================================================

class OwnerConsole:
    """Interactive owner console for runtime management"""
    
    def __init__(self, application: Application):
        self.application = application
        self.running = False
    
    def start(self):
        """Start console in background thread"""
        if not sys.stdin.isatty() or not GLOBAL_SETTINGS.get('owner_console_enabled', True):
            logger.info("Owner console disabled or not running in TTY.")
            return
        
        self.running = True
        console_thread = threading.Thread(target=self._console_loop, daemon=True)
        console_thread.start()
        logger.info("Owner console started. Type 'help' for commands.")
    
    def _console_loop(self):
        """Main console loop"""
        print("\n" + "="*50)
        print("🚀 ForwardBot Owner Console")
        print("Type 'help' for available commands")
        print("="*50 + "\n")
        
        while self.running:
            try:
                command = input("owner> ").strip().lower()
                
                if not command:
                    continue
                
                if command == "quit" or command == "exit":
                    print("👋 Goodbye!")
                    os._exit(0)
                
                elif command == "help":
                    self._show_help()
                
                elif command == "stats":
                    self._show_stats()
                
                elif command == "users":
                    self._show_users()
                
                elif command == "rules":
                    self._show_rules()
                
                elif command == "reload":
                    forwarding_engine.reload_rules()
                    GLOBAL_SETTINGS.update(db.get_all_global_settings()) # Reload global settings
                    print("✅ Rules and global settings reloaded")
                
                elif command == "save":
                    # Global settings are saved immediately by db.set_global_setting
                    print("✅ Data saved (global settings are auto-saved)")
                
                elif command.startswith("set"):
                    parts = command.split(maxsplit=2)
                    if len(parts) == 3:
                        key = parts[1]
                        value_str = parts[2]
                        try:
                            value = json.loads(value_str) # Try to parse as JSON (bool, int, list, dict)
                        except json.JSONDecodeError:
                            value = value_str # Treat as string if not JSON
                        
                        if key in GLOBAL_SETTINGS:
                            db.set_global_setting(key, value)
                            GLOBAL_SETTINGS[key] = value # Update in-memory cache
                            print(f"✅ Setting '{key}' updated to '{value}'")
                            if key == "rate_limit":
                                self.application.rate_limiter.overall_max_rate = value
                                print("✅ Application rate limiter updated.")
                            elif key == "maintenance_mode":
                                forwarding_engine.reload_rules() # Rules are paused in maintenance mode
                        else:
                            print(f"❌ Unknown setting: {key}")
                    else:
                        print("Usage: set <key> <value>")
                        print("Example: set maintenance_mode true")
                        print("Example: set rate_limit 100")
                        print("Example: set max_rules_per_user 10")
                
                elif command.startswith("get"):
                    parts = command.split()
                    if len(parts) == 2:
                        key = parts[1]
                        value = GLOBAL_SETTINGS.get(key, "Not Found")
                        print(f"Setting '{key}': {value}")
                    else:
                        print("Usage: get <key>")
                
                elif command.startswith("promote"):
                    parts = command.split()
                    if len(parts) == 3:
                        try:
                            user_id = int(parts[1])
                            role = parts[2]
                            if RoleManager.promote_user(user_id, role):
                                print(f"✅ User {user_id} promoted to {role}")
                            else:
                                print(f"❌ Failed to promote user {user_id}")
                        except ValueError:
                            print("❌ Invalid user ID")
                    else:
                        print("Usage: promote <user_id> <role>")
                
                elif command == "backup":
                    self._create_backup_sync()
                
                elif command == "status":
                    self._show_status()
                
                elif command == "clear":
                    os.system('cls' if os.name == 'nt' else 'clear')
                
                else:
                    print(f"❌ Unknown command: {command}")
                    print("Type 'help' for available commands")
            
            except KeyboardInterrupt:
                print("\n👋 Console interrupted. Use 'quit' to exit properly.")
            except EOFError:
                print("\n👋 Console closed.")
                break
            except Exception as e:
                print(f"❌ Console error: {e}")
                traceback.print_exc()
    
    def _show_help(self):
        """Show console help"""
        help_text = """
🚀 Owner Console Commands:

📊 Information:
  stats          - Show bot statistics
  users          - Show user overview  
  rules          - Show rules overview
  status         - Show system status

⚙️ Management:
  set <key> <value>         - Set a global bot setting (e.g., `set maintenance_mode true`)
  get <key>                 - Get a global bot setting
  promote <id> <role>       - Promote user
  reload         - Reload forwarding rules and global settings
  backup         - Create system backup

🔧 Utility:
  clear          - Clear console screen
  help           - Show this help
  quit/exit      - Exit console and bot
"""
        print(help_text)
    
    def _show_stats(self):
        """Show statistics"""
        stats = forwarding_engine.stats_in_memory
        print(f"""
📊 Bot Statistics (In-memory since last restart):
  Total forwards: {stats['total_forwards']:,}
  Successful: {stats['successful_forwards']:,}
  Failed: {stats['failed_forwards']:,}
  Rate limited: {stats['rate_limited']:,}
  Active rules: {len(forwarding_engine.active_rules)}
""")
        # Also show aggregated DB stats
        db_stats = db.get_statistics()
        print(f"""
📊 Bot Statistics (Aggregated from Database):
  Total forwards (DB): {db_stats.total_forwards:,}
  Successful (DB): {db_stats.successful_forwards:,}
  Failed (DB): {db_stats.failed_forwards:,}
  Total users (DB): {db_stats.total_users:,}
  Total rules (DB): {db_stats.total_rules:,}
""")
    
    def _show_users(self):
        """Show user overview"""
        all_users = db.get_all_users()
        total_users = len(all_users)
        premium_users = len([u for u in all_users if u.role == 'premium'])
        admin_users = len([u for u in all_users if u.role == 'admin'])
        banned_users = len([u for u in all_users if u.is_banned])
        
        print(f"""
👥 User Overview:
  Total users: {total_users}
  Premium users: {premium_users}
  Admin users: {admin_users}
  Free users: {total_users - premium_users - admin_users}
  Banned users: {banned_users}
""")
    
    def _show_rules(self):
        """Show rules overview"""
        all_rules = db.get_rules()
        active_rules = len([r for r in all_rules if r.is_active])
        
        print(f"""
📝 Rules Overview:
  Total rules: {len(all_rules)}
  Active rules: {active_rules}
  Inactive rules: {len(all_rules) - active_rules}
  Active sources: {len(forwarding_engine.active_rules)}
""")
    
    def _show_status(self):
        """Show system status"""
        uptime = int(time.time() - self.application.bot_data.get('start_time', time.time()))
        
        print(f"""
🔧 System Status:
  Uptime: {format_duration(uptime)}
  Maintenance: {'ON' if GLOBAL_SETTINGS.get('maintenance_mode') else 'OFF'}
  Phone required: {'ON' if GLOBAL_SETTINGS.get('phone_required') else 'OFF'}
  Auto forward: {'ON' if GLOBAL_SETTINGS.get('auto_forward') else 'OFF'}
  Database size: {os.path.getsize(DATABASE_URL) / 1024:.1f} KB
  Owner Console Enabled: {'ON' if GLOBAL_SETTINGS.get('owner_console_enabled') else 'OFF'}
""")
    
    def _create_backup_sync(self) -> Tuple[str, str]:
        """Create system backup (synchronous version for console/callbacks)"""
        backup_name = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        db_backup_name = f"db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        backup_path = BACKUP_DIR / backup_name
        db_backup_path = BACKUP_DIR / db_backup_name
        
        try:
            # Copy SQLite database file
            shutil.copyfile(DATABASE_URL, db_backup_path)
            
            # Save global settings and other metadata
            backup_data = {
                "timestamp": get_current_time(),
                "version": "3.0.0",
                "global_settings": GLOBAL_SETTINGS,
                "database_backup_file": str(db_backup_path.name)
            }
            
            backup_path.write_text(
                json.dumps(backup_data, indent=2, default=str),
                encoding='utf-8'
            )
            
            print(f"✅ Backup created: {backup_name} and {db_backup_name}")
            return backup_name, db_backup_name
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            raise # Re-raise for error handling in calling function

# ============================================================================
# ERROR HANDLERS
# ============================================================================

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and send a telegram message to notify the developer."""
    logger.error(f"Exception while handling an update: {context.error}")
    
    # Log to error file
    error_logger.error(
        f"Update: {update}\nError: {context.error}",
        exc_info=context.error
    )
    
    try:
        user_info = "N/A"
        if hasattr(update, 'effective_user') and update.effective_user:
            user_info = f"User ID: {update.effective_user.id}, Name: {update.effective_user.full_name}"
        elif hasattr(update, 'callback_query') and update.callback_query and update.callback_query.from_user:
            user_info = f"User ID: {update.callback_query.from_user.id}, Name: {update.callback_query.from_user.full_name}"

        update_str = str(update)
        if len(update_str) > 1000:
            update_str = update_str[:997] + "..."

        await context.bot.send_message(
            chat_id=OWNER_ID,
            text=f"🚨 <b>Bot Error Detected!</b> \n\n" # Changed to HTML
                 f"<b>Error:</b> <code>{escape_html(str(context.error))}</code>\n" # Changed to HTML
                 f"<b>Time:</b> {escape_html(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}\n" # Changed to HTML
                 f"<b>User:</b> {escape_html(user_info)}\n" # Changed to HTML
                 f"<b>Update:</b> <code>{escape_html(update_str)}</code>\n\n" # Changed to HTML
                 f"Check logs for full traceback.",
            parse_mode=ParseMode.HTML # Changed to HTML
        )
    except Exception as e:
        logger.error(f"Failed to send error notification to owner: {e}")
        pass
    
    if update and hasattr(update, 'effective_message') and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Oops! Something went wrong. The developers have been notified."
            )
        except Exception as e:
            logger.error(f"Failed to send user error message: {e}")
            pass
    elif update and hasattr(update, 'callback_query') and update.callback_query:
        try:
            await update.callback_query.answer("Oops! Something went wrong.", show_alert=True)
        except Exception as e:
            logger.error(f"Failed to answer callback query with error: {e}")
            pass

# ============================================================================
# SCHEDULED JOBS
# ============================================================================

async def daily_cleanup_job(context: ContextTypes.DEFAULT_TYPE):
    """Daily cleanup job"""
    logger.info("Running daily cleanup...")
    
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Aggregate in-memory stats for the day and reset
        daily_forwards_data = {
            'total': forwarding_engine.stats_in_memory['total_forwards'],
            'successful': forwarding_engine.stats_in_memory['successful_forwards'],
            'failed': forwarding_engine.stats_in_memory['failed_forwards']
        }
        forwarding_engine.stats_in_memory = {
            'total_forwards': 0, 'successful_forwards': 0, 'failed_forwards': 0, 'rate_limited': 0
        }

        # Get current user and rule counts for daily snapshot
        total_users_today = len(db.get_all_users())
        total_rules_today = len(db.get_rules())

        # This is a simplification. For 'users_joined' and 'rules_created'
        # you'd need to track these specifically during user/rule creation.
        # For now, we'll just pass 0 or rely on the DB's daily_statistics table
        # to accumulate these if they were logged there.
        # The `db.update_daily_statistics` function now takes these.
        # The `start_command` already logs new users. Rule creation should also log.
        
        db.update_daily_statistics(today, daily_forwards_data, {
            'users_joined': 0, # This should be tracked by start_command
            'rules_created': 0 # This should be tracked by autoforward_command
        })
        logger.info(f"Daily statistics updated for {today}.")

        # Clean old logs (keep last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        with sqlite3.connect(DATABASE_URL) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM logs WHERE timestamp < ?", (cutoff_date.isoformat(),))
            conn.commit()
        logger.info("Old logs cleaned.")
        
        # Create daily backup if enabled
        if GLOBAL_SETTINGS.get('backup_enabled', True):
            console = OwnerConsole(context.application)
            try:
                console._create_backup_sync()
            except Exception as e:
                logger.error(f"Automated daily backup failed: {e}")
        
        logger.info("Daily cleanup completed")
        
    except Exception as e:
        logger.error(f"Daily cleanup failed: {e}")

async def hourly_stats_job(context: ContextTypes.DEFAULT_TYPE):
    """Hourly statistics update"""
    try:
        logger.debug("Running hourly stats update...")
        # This job can be used to aggregate more granular stats or push to external monitoring
        # For now, it just logs.
    except Exception as e:
        logger.error(f"Hourly stats job failed: {e}")

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
       """Handle all non-command messages for auto-forwarding"""
    
       # Debug logging
       logger.debug(f"Update type: {type(update)}")
       logger.debug(f"Has message: {update.message is not None}")
       logger.debug(f"Has channel_post: {update.channel_post is not None}")
       logger.debug(f"Has effective_user: {update.effective_user is not None}")
       logger.debug(f"Has effective_chat: {update.effective_chat is not None}")
    
# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main function to run the bot"""
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN environment variable is required!")
        sys.exit(1)
    
    if not OWNER_ID:
        logger.error("OWNER_ID environment variable is required!")
        sys.exit(1)
    
    logger.info("🚀 Starting ForwardBot Premium v3.0.0...")
    
    defaults = Defaults(
        parse_mode=ParseMode.HTML, # Changed to HTML for consistency
        block=False,
        quote=True
    )
    
    application = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .defaults(defaults)
        .rate_limiter(AIORateLimiter(overall_max_rate=GLOBAL_SETTINGS.get('rate_limit', RATE_LIMIT)))
        .build()
    )
    
    # Store bot start time
    application.bot_data['start_time'] = time.time()

    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("connect_phone", connect_phone_command))
    application.add_handler(CommandHandler("forward", forward_command))
    application.add_handler(CommandHandler("rules", rules_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("logs", logs_command))
    application.add_handler(CommandHandler("settings", user_settings_command)) # User settings

    # Admin commands
    application.add_handler(CommandHandler("users", users_command))
    application.add_handler(CommandHandler("promote", promote_command))
    application.add_handler(CommandHandler("demote", demote_command))
    application.add_handler(CommandHandler("ban", ban_command))
    application.add_handler(CommandHandler("unban", unban_command))
    application.add_handler(CommandHandler("broadcast", broadcast_command))
    application.add_handler(CommandHandler("system_stats", system_stats_command))
    application.add_handler(CommandHandler("bot_settings", bot_settings_command)) # Global bot settings
    application.add_handler(MessageHandler(filters.CONTACT, contact_handler))
    # Owner commands  
    application.add_handler(CommandHandler("maintenance", maintenance_command))
    application.add_handler(CommandHandler("backup", backup_command))
    application.add_handler(CommandHandler("restore", restore_backup_menu)) # Changed to menu
    application.add_handler(CommandHandler("debug", debug_command))
    application.add_handler(CommandHandler("clear_logs", clear_logs_command))
    
    # Add this before application.run_polling() in main()
    application.add_error_handler(error_handler)

    # Conversation Handlers
    # Rule Creation
    application.add_handler(ConversationHandler(
        entry_points=[CommandHandler("autoforward", autoforward_command)],
        states={
            CREATE_RULE_MODE: [CallbackQueryHandler(create_rule_mode, pattern="^mode_")],
            CREATE_RULE_SOURCE: [MessageHandler(filters.TEXT & ~filters.COMMAND, create_rule_source)],
            CREATE_RULE_TARGETS: [MessageHandler(filters.TEXT & ~filters.COMMAND, create_rule_targets)],
        },
        fallbacks=[CallbackQueryHandler(cancel_conversation, pattern="^cancel_conversation$"),
                   CommandHandler("cancel", cancel_conversation)],
        # Removed map_to_parent as this is a top-level conversation
    ))

    # Rule Editing
    application.add_handler(ConversationHandler(
        entry_points=[CallbackQueryHandler(edit_rule_start, pattern="^rule_settings_")], # Entry point for rule settings
        states={
            EDIT_RULE_SELECT: [CallbackQueryHandler(edit_rule_callback_handler, pattern="^edit_rule_.*|^confirm_delete_rule_.*|^set_mode_.*")],
            EDIT_RULE_KEYWORDS: [MessageHandler(filters.TEXT & ~filters.COMMAND, edit_rule_text_input)],
            EDIT_RULE_EXCLUDE_KEYWORDS: [MessageHandler(filters.TEXT & ~filters.COMMAND, edit_rule_text_input)],
            EDIT_RULE_REPLACE_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, edit_rule_text_input)],
            EDIT_RULE_FILTERS: [CallbackQueryHandler(edit_rule_callback_handler, pattern="^edit_rule_filter_.*")],
            EDIT_RULE_SCHEDULE: [CallbackQueryHandler(edit_rule_callback_handler, pattern="^edit_rule_schedule_.*")],
            EDIT_RULE_TARGETS: [MessageHandler(filters.TEXT & ~filters.COMMAND, edit_rule_text_input)],
            EDIT_RULE_MODE: [CallbackQueryHandler(edit_rule_callback_handler, pattern="^set_mode_.*")],
            EDIT_RULE_MEDIA_TYPES: [MessageHandler(filters.TEXT & ~filters.COMMAND, edit_rule_text_input)],
            EDIT_RULE_LENGTH_FILTERS: [MessageHandler(filters.TEXT & ~filters.COMMAND, edit_rule_text_input)],
            EDIT_RULE_SCHEDULE_TIME: [MessageHandler(filters.TEXT & ~filters.COMMAND, edit_rule_text_input)],
            EDIT_RULE_SCHEDULE_DAYS: [MessageHandler(filters.TEXT & ~filters.COMMAND, edit_rule_text_input)],
            SETTING_CHANGE_VALUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_change_value)], # For user settings text input
        },
        fallbacks=[CallbackQueryHandler(cancel_conversation, pattern="^cancel_conversation$"),
                   CommandHandler("cancel", cancel_conversation)],
        # Removed map_to_parent as this is a top-level conversation
    ))

    # Broadcast Conversation (already defined as broadcast_handler)
    application.add_handler(broadcast_handler)

    # Message Handler for all non-command messages
    application.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, message_handler)) # Ensure it doesn't catch commands
    
       # Rest of the function...

    # Callback Query Handler for inline buttons (general ones not part of conversations)
    application.add_handler(CallbackQueryHandler(callback_query_handler, pattern="^(?!mode_|edit_rule_|confirm_delete_rule_|set_mode_|broadcast_).*")) # Exclude patterns handled by conversations
    application.add_handler(CallbackQueryHandler(view_user_profile_callback, pattern="^view_user_profile_")) # Specific handler for user profile view

    # Scheduled Jobs
    job_queue = application.job_queue
    job_queue.run_daily(daily_cleanup_job, time=dtime(hour=2, minute=0))
    job_queue.run_repeating(hourly_stats_job, interval=3600, first=0)

    # Start the bot
    application.run_polling()

if __name__ == "__main__":
    main()
