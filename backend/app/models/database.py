"""
MongoDB database connection and operations
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, OperationFailure
from loguru import logger

from backend.app.config import settings
from .schemas import AttackResult, TargetModelResponse, JudgeVerdict
from .enums import VerdictType


class Database:
    """MongoDB database manager"""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        
        # Existing collections
        self.attacks_collection = None
        self.responses_collection = None
        self.evaluations_collection = None
        self.metrics_collection = None
        
        # Multi-turn collections
        self.conversations_collection = None
        self.multiturn_results_collection = None

        # NEW: Audit session collection (replaces session_state storage)
        self.audit_sessions_collection = None

    async def connect(self):
        """Establish database connection"""
        try:
            if self.client is not None:
                self.client.close()
                self.client = None

            self.client = AsyncIOMotorClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=5000
            )
            self.db = self.client[settings.MONGODB_DB_NAME]
            
            # Test connection
            await self.client.admin.command('ping')
            
            # Initialize collection references
            self.attacks_collection = self.db["attacks"]
            self.responses_collection = self.db["responses"]
            self.evaluations_collection = self.db["evaluations"]
            self.metrics_collection = self.db["metrics"]
            
            # Multi-turn collections
            self.conversations_collection = self.db["conversations"]
            self.multiturn_results_collection = self.db["multiturn_results"]

            # NEW: Audit sessions
            self.audit_sessions_collection = self.db["audit_sessions"]
            
            logger.info(f"✅ Connected to MongoDB: {settings.MONGODB_DB_NAME}")
            
            await self._create_indexes()
            
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            self.client = None
            logger.info("🔌 Disconnected from MongoDB")
    
    async def _create_indexes(self):
        """Create database indexes for performance"""
        try:
            # Attacks collection
            await self.db.attacks.create_index("attack_id", unique=True)
            await self.db.attacks.create_index("timestamp")
            await self.db.attacks.create_index("strategy_type")
            
            # Responses collection
            await self.db.responses.create_index("response_id", unique=True)
            await self.db.responses.create_index("attack_id")
            await self.db.responses.create_index("timestamp")
            
            # Evaluations collection
            await self.db.evaluations.create_index("evaluation_id", unique=True)
            await self.db.evaluations.create_index("response_id")
            await self.db.evaluations.create_index("verdict")
            await self.db.evaluations.create_index("timestamp")
            
            # Multi-turn conversations indexes
            await self.conversations_collection.create_index([
                ("conversation_id", 1),
                ("turn_number", 1)
            ])
            await self.conversations_collection.create_index("timestamp")
            await self.conversations_collection.create_index("verdict")
            
            # Multi-turn results indexes
            await self.multiturn_results_collection.create_index(
                "conversation_id", unique=True
            )
            await self.multiturn_results_collection.create_index("jailbreak_achieved")
            await self.multiturn_results_collection.create_index("timestamp")
            await self.multiturn_results_collection.create_index("target_model")

            # NEW: Audit sessions indexes
            await self.audit_sessions_collection.create_index("session_id", unique=True)
            await self.audit_sessions_collection.create_index("timestamp")
            await self.audit_sessions_collection.create_index("model")
            
            logger.info("✅ Database indexes created (including audit_sessions)")
            
        except OperationFailure as e:
            logger.warning(f"⚠️ Failed to create indexes: {e}")
    
    # ========================================================================
    # INSERT OPERATIONS
    # ========================================================================
    
    async def insert_attack(self, attack: AttackResult) -> str:
        """Insert attack record"""
        try:
            doc = attack.model_dump()
            await self.db.attacks.insert_one(doc)
            logger.debug(f"✅ Inserted attack: {attack.attack_id}")
            return attack.attack_id
        except Exception as e:
            logger.error(f"❌ Failed to insert attack: {e}")
            raise
    
    async def insert_response(self, response: TargetModelResponse) -> str:
        """Insert response record"""
        try:
            doc = response.model_dump()
            await self.db.responses.insert_one(doc)
            logger.debug(f"✅ Inserted response: {response.response_id}")
            return response.response_id
        except Exception as e:
            logger.error(f"❌ Failed to insert response: {e}")
            raise
    
    async def insert_evaluation(self, evaluation: JudgeVerdict) -> str:
        """Insert evaluation record"""
        try:
            doc = evaluation.model_dump()
            await self.db.evaluations.insert_one(doc)
            logger.debug(f"✅ Inserted evaluation: {evaluation.evaluation_id}")
            return evaluation.evaluation_id
        except Exception as e:
            logger.error(f"❌ Failed to insert evaluation: {e}")
            raise
    
    # ========================================================================
    # QUERY OPERATIONS
    # ========================================================================
    
    async def get_attack_by_id(self, attack_id: str) -> Optional[Dict[str, Any]]:
        """Get attack by ID"""
        try:
            return await self.db.attacks.find_one({"attack_id": attack_id})
        except Exception as e:
            logger.error(f"❌ Failed to get attack: {e}")
            return None
    
    async def get_response_by_attack_id(self, attack_id: str) -> Optional[Dict[str, Any]]:
        """Get response by attack ID"""
        try:
            return await self.db.responses.find_one({"attack_id": attack_id})
        except Exception as e:
            logger.error(f"❌ Failed to get response: {e}")
            return None
    
    async def get_evaluation_by_response_id(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Get evaluation by response ID"""
        try:
            return await self.db.evaluations.find_one({"response_id": response_id})
        except Exception as e:
            logger.error(f"❌ Failed to get evaluation: {e}")
            return None
    
    async def get_evaluations_by_filter(
        self,
        verdict: Optional[VerdictType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get evaluations with filters"""
        try:
            query = {}
            if verdict:
                query["verdict"] = verdict.value
            if start_date or end_date:
                query["timestamp"] = {}
                if start_date:
                    query["timestamp"]["$gte"] = start_date
                if end_date:
                    query["timestamp"]["$lte"] = end_date
            cursor = self.db.evaluations.find(query).limit(limit)
            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.error(f"❌ Failed to get evaluations: {e}")
            return []
    
    # ========================================================================
    # ANALYTICS OPERATIONS
    # ========================================================================
    
    async def calculate_asr(
        self,
        target_model: Optional[str] = None,
        strategy_name: Optional[str] = None,
        days: int = 30
    ) -> float:
        """Calculate Attack Success Rate"""
        try:
            pipeline = []
            match_stage = {
                "timestamp": {"$gte": datetime.utcnow() - timedelta(days=days)}
            }
            if target_model:
                match_stage["target_model"] = target_model
            pipeline.append({"$match": match_stage})
            pipeline.append({
                "$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "jailbroken": {
                        "$sum": {
                            "$cond": [{"$eq": ["$verdict", VerdictType.JAILBROKEN.value]}, 1, 0]
                        }
                    }
                }
            })
            pipeline.append({
                "$project": {
                    "asr": {
                        "$cond": [
                            {"$eq": ["$total", 0]}, 0,
                            {"$divide": ["$jailbroken", "$total"]}
                        ]
                    }
                }
            })
            cursor = self.db.evaluations.aggregate(pipeline)
            results = await cursor.to_list(length=1)
            return results[0].get("asr", 0.0) if results else 0.0
        except Exception as e:
            logger.error(f"❌ Failed to calculate ASR: {e}")
            return 0.0
    
    async def get_collection_count(self, collection_name: str) -> int:
        """Get document count for collection"""
        try:
            return await self.db[collection_name].count_documents({})
        except Exception as e:
            logger.error(f"❌ Failed to get count: {e}")
            return 0
    
    # ========================================================================
    # MULTI-TURN OPERATIONS
    # ========================================================================
    
    async def insert_conversation_turn(
        self,
        conversation_id: str,
        turn_data: Dict[str, Any]
    ) -> str:
        """Insert a single conversation turn"""
        try:
            doc = {
                "conversation_id": conversation_id,
                **turn_data,
                "timestamp": datetime.utcnow()
            }
            result = await self.conversations_collection.insert_one(doc)
            logger.debug(f"✅ Inserted turn for conversation: {conversation_id[:8]}...")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"❌ Failed to insert conversation turn: {e}")
            raise
    
    async def insert_multiturn_result(self, result_data: Dict[str, Any]) -> str:
        """Insert complete multi-turn conversation result"""
        try:
            doc = {**result_data, "timestamp": datetime.utcnow()}
            result = await self.multiturn_results_collection.insert_one(doc)
            logger.debug(f"✅ Inserted multi-turn result: {result_data.get('conversation_id', 'unknown')[:8]}...")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"❌ Failed to insert multi-turn result: {e}")
            raise
    
    async def get_conversation_turns(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all turns for a conversation"""
        try:
            cursor = self.conversations_collection.find(
                {"conversation_id": conversation_id}
            ).sort("turn_number", 1)
            return await cursor.to_list(length=100)
        except Exception as e:
            logger.error(f"❌ Failed to get conversation turns: {e}")
            return []
    
    async def get_multiturn_result(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get complete multi-turn result"""
        try:
            return await self.multiturn_results_collection.find_one(
                {"conversation_id": conversation_id}
            )
        except Exception as e:
            logger.error(f"❌ Failed to get multi-turn result: {e}")
            return None
    
    async def get_multiturn_history(
        self,
        limit: int = 20,
        skip: int = 0,
        jailbroken_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get multi-turn conversation history"""
        try:
            query = {"jailbreak_achieved": True} if jailbroken_only else {}
            cursor = self.multiturn_results_collection.find(query)\
                .sort("timestamp", -1).skip(skip).limit(limit)
            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.error(f"❌ Failed to get multi-turn history: {e}")
            return []
    
    async def calculate_multiturn_asr(
        self,
        target_model: Optional[str] = None,
        days: int = 30
    ) -> float:
        """Calculate Multi-Turn Attack Success Rate"""
        try:
            pipeline = []
            match_stage = {
                "timestamp": {"$gte": datetime.utcnow() - timedelta(days=days)}
            }
            if target_model:
                match_stage["target_model"] = target_model
            pipeline.append({"$match": match_stage})
            pipeline.append({
                "$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "jailbroken": {
                        "$sum": {"$cond": [{"$eq": ["$jailbreak_achieved", True]}, 1, 0]}
                    }
                }
            })
            pipeline.append({
                "$project": {
                    "asr": {
                        "$cond": [
                            {"$eq": ["$total", 0]}, 0,
                            {"$divide": ["$jailbroken", "$total"]}
                        ]
                    }
                }
            })
            cursor = self.multiturn_results_collection.aggregate(pipeline)
            results = await cursor.to_list(length=1)
            return results[0].get("asr", 0.0) if results else 0.0
        except Exception as e:
            logger.error(f"❌ Failed to calculate multi-turn ASR: {e}")
            return 0.0
    
    async def get_strategy_effectiveness(self, days: int = 30) -> Dict[str, float]:
        """Get strategy effectiveness across all multi-turn conversations"""
        try:
            pipeline = [
                {"$match": {"timestamp": {"$gte": datetime.utcnow() - timedelta(days=days)}}},
                {"$unwind": "$strategies_tried"},
                {
                    "$group": {
                        "_id": "$strategies_tried",
                        "total_uses": {"$sum": 1},
                        "successes": {
                            "$sum": {"$cond": [{"$eq": ["$jailbreak_achieved", True]}, 1, 0]}
                        }
                    }
                },
                {
                    "$project": {
                        "strategy": "$_id",
                        "success_rate": {
                            "$cond": [
                                {"$eq": ["$total_uses", 0]}, 0,
                                {"$divide": ["$successes", "$total_uses"]}
                            ]
                        }
                    }
                }
            ]
            cursor = self.multiturn_results_collection.aggregate(pipeline)
            results = await cursor.to_list(length=100)
            return {r.get("strategy", "unknown"): r.get("success_rate", 0.0) for r in results}
        except Exception as e:
            logger.error(f"❌ Failed to get strategy effectiveness: {e}")
            return {}

    # ========================================================================
    # NEW: AUDIT SESSION OPERATIONS
    # ========================================================================

    async def save_audit_session(
        self,
        examples: List[Dict[str, Any]],
        model: str,
        forbidden_goal: str,
    ) -> str:
        """
        Persist a full audit session (e.g. 19 rows from run_full_audit) to MongoDB.

        Stores the raw rows so the retraining page can query by session_id
        instead of pulling all historical data.

        Returns:
            session_id  — a UUID string the caller should store in st.session_state
        """
        try:
            session_id = str(uuid.uuid4())
            jailbroken = sum(
                1 for e in examples
                if (e.get("verdict") or "").upper() in ("JAILBROKEN", "PARTIAL")
            )
            refused = sum(
                1 for e in examples
                if (e.get("verdict") or "").upper() == "REFUSED"
            )
            doc = {
                "session_id":     session_id,
                "model":          model,
                "forbidden_goal": forbidden_goal,
                "timestamp":      datetime.utcnow(),
                "examples":       examples,
                "total":          len(examples),
                "jailbroken":     jailbroken,
                "refused":        refused,
            }
            await self.audit_sessions_collection.insert_one(doc)
            logger.info(
                "✅ Audit session saved | session_id=%s model=%s jailbroken=%d refused=%d",
                session_id, model, jailbroken, refused,
            )
            return session_id
        except Exception as e:
            logger.error(f"❌ Failed to save audit session: {e}")
            raise

    async def load_audit_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a saved audit session by its session_id.

        Returns the full document including the `examples` list,
        or None if not found.
        """
        try:
            result = await self.audit_sessions_collection.find_one(
                {"session_id": session_id},
                {"_id": 0}   # exclude Mongo internal _id from result
            )
            if not result:
                logger.warning(f"⚠️ Audit session not found: {session_id}")
            return result
        except Exception as e:
            logger.error(f"❌ Failed to load audit session: {e}")
            return None
        
    
    async def list_audit_sessions(
        self,
        limit: int = 20,
        model_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List recent audit sessions for the retraining UI dropdown.

        Returns lightweight summaries (no `examples` array) sorted newest first.

        Args:
            limit:        Max number of sessions to return.
            model_filter: Optional — filter by model name (e.g. "ollama/phi3:latest").
        """
        try:
            query: Dict[str, Any] = {}
            if model_filter:
                query["model"] = model_filter

            cursor = self.audit_sessions_collection.find(
                query,
                # Projection — exclude the heavy examples array for the list view
                {
                    "_id":           0,
                    "session_id":    1,
                    "model":         1,
                    "forbidden_goal": 1,
                    "timestamp":     1,
                    "total":         1,
                    "jailbroken":    1,
                    "refused":       1,
                }
            ).sort("timestamp", -1).limit(limit)

            results = await cursor.to_list(length=limit)
            logger.debug(f"Listed {len(results)} audit sessions")
            return results
        except Exception as e:
            logger.error(f"❌ Failed to list audit sessions: {e}")
            return []
    async def update_or_create_batch_session(self, examples: List[Dict[str, Any]], model: str, forbidden_goal: str, session_id: Optional[str] = None) -> str:
        try:
            if session_id:
                # Append multiple rows to the existing session
                jailbroken_count = sum(1 for e in examples if e.get("verdict") in ("JAILBROKEN", "PARTIAL"))
                refused_count = sum(1 for e in examples if e.get("verdict") == "REFUSED")
                
                await self.audit_sessions_collection.update_one(
                    {"session_id": session_id},
                    {
                        "$push": {"examples": {"$each": examples}}, # $each allows appending a list
                        "$inc": {
                            "total": len(examples),
                            "jailbroken": jailbroken_count,
                            "refused": refused_count
                        },
                        "$set": {"timestamp": datetime.utcnow()}
                    }
                )
                return session_id
            else:
                # Create fresh session
                return await self.save_audit_session(examples, model, forbidden_goal)
        except Exception as e:
            logger.error(f"❌ Batch session update failed: {e}")
            raise

    async def delete_audit_session(self, session_id: str) -> bool:
        """
        Delete an audit session by ID.
        Useful for cleanup or if the user wants to remove old sessions.

        Returns True if a document was deleted, False if not found.
        """
        try:
            result = await self.audit_sessions_collection.delete_one(
                {"session_id": session_id}
            )
            deleted = result.deleted_count > 0
            if deleted:
                logger.info(f"🗑️  Deleted audit session: {session_id}")
            else:
                logger.warning(f"⚠️  Audit session not found for deletion: {session_id}")
            return deleted
        except Exception as e:
            logger.error(f"❌ Failed to delete audit session: {e}")
            return False
    async def update_or_create_session(
    self,
    example: Dict[str, Any],
    model: str,
    forbidden_goal: str,
    session_id: Optional[str] = None
    ) -> str:
        """
        If session_id exists, append the example to the array.
        If not, create a new session document.
        """
        try:
            if session_id:
                # 1. Update existing session by pushing to the 'examples' array
                await self.audit_sessions_collection.update_one(
                    {"session_id": session_id},
                    {
                        "$push": {"examples": example},
                        "$inc": {
                            "total": 1,
                            "jailbroken": 1 if example.get("verdict") in ("JAILBROKEN", "PARTIAL") else 0,
                            "refused": 1 if example.get("verdict") == "REFUSED" else 0
                        },
                        "$set": {"timestamp": datetime.utcnow()} # Update last activity
                    }
                )
                return session_id
            else:
                # 2. No session exists, create a new one using your existing logic
                return await self.save_audit_session([example], model, forbidden_goal)
                
        except Exception as e:
            logger.error(f"❌ Failed to update/create session: {e}")
            raise

# Global database instance
db = Database()


async def get_database() -> Database:
    """Get database instance (for FastAPI dependency injection)"""
    return db