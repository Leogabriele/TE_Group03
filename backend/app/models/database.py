"""
MongoDB database connection and operations
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
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
        
        # Existing collections (will be initialized in connect)
        self.attacks_collection = None
        self.responses_collection = None
        self.evaluations_collection = None
        self.metrics_collection = None
        
        # NEW: Multi-turn collections
        self.conversations_collection = None  # Individual turns
        self.multiturn_results_collection = None  # Complete conversations

    async def connect(self):
        """Establish database connection"""
        try:
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
            
            # NEW: Multi-turn collections
            self.conversations_collection = self.db["conversations"]
            self.multiturn_results_collection = self.db["multiturn_results"]
            
            logger.info(f"✅ Connected to MongoDB: {settings.MONGODB_DB_NAME}")
            
            # Create indexes
            await self._create_indexes()
            
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("🔌 Disconnected from MongoDB")
    
    async def _create_indexes(self):
        """Create database indexes for performance"""
        try:
            # Existing indexes - Attacks collection
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
            
            # NEW: Multi-turn conversations indexes
            await self.conversations_collection.create_index([
                ("conversation_id", 1),
                ("turn_number", 1)
            ])
            await self.conversations_collection.create_index("timestamp")
            await self.conversations_collection.create_index("verdict")
            
            # Multi-turn results indexes
            await self.multiturn_results_collection.create_index(
                "conversation_id", 
                unique=True
            )
            await self.multiturn_results_collection.create_index("jailbreak_achieved")
            await self.multiturn_results_collection.create_index("timestamp")
            await self.multiturn_results_collection.create_index("target_model")
            
            logger.info("✅ Database indexes created (including multi-turn)")
            
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
            result = await self.db.attacks.find_one({"attack_id": attack_id})
            return result
        except Exception as e:
            logger.error(f"❌ Failed to get attack: {e}")
            return None
    
    async def get_response_by_attack_id(self, attack_id: str) -> Optional[Dict[str, Any]]:
        """Get response by attack ID"""
        try:
            result = await self.db.responses.find_one({"attack_id": attack_id})
            return result
        except Exception as e:
            logger.error(f"❌ Failed to get response: {e}")
            return None
    
    async def get_evaluation_by_response_id(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Get evaluation by response ID"""
        try:
            result = await self.db.evaluations.find_one({"response_id": response_id})
            return result
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
            results = await cursor.to_list(length=limit)
            return results
            
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
            
            # Match stage
            match_stage = {
                "timestamp": {
                    "$gte": datetime.utcnow() - timedelta(days=days)
                }
            }
            
            if target_model:
                match_stage["target_model"] = target_model
            
            pipeline.append({"$match": match_stage})
            
            # Group and calculate
            pipeline.append({
                "$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "jailbroken": {
                        "$sum": {
                            "$cond": [
                                {"$eq": ["$verdict", VerdictType.JAILBROKEN.value]},
                                1,
                                0
                            ]
                        }
                    }
                }
            })
            
            # Project ASR
            pipeline.append({
                "$project": {
                    "asr": {
                        "$cond": [
                            {"$eq": ["$total", 0]},
                            0,
                            {"$divide": ["$jailbroken", "$total"]}
                        ]
                    }
                }
            })
            
            cursor = self.db.evaluations.aggregate(pipeline)
            results = await cursor.to_list(length=1)
            
            if results:
                return results[0].get("asr", 0.0)
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate ASR: {e}")
            return 0.0
    
    async def get_collection_count(self, collection_name: str) -> int:
        """Get document count for collection"""
        try:
            count = await self.db[collection_name].count_documents({})
            return count
        except Exception as e:
            logger.error(f"❌ Failed to get count: {e}")
            return 0
    
    # ========================================================================
    # NEW: MULTI-TURN OPERATIONS
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
    
    async def insert_multiturn_result(
        self, 
        result_data: Dict[str, Any]
    ) -> str:
        """Insert complete multi-turn conversation result"""
        try:
            doc = {
                **result_data,
                "timestamp": datetime.utcnow()
            }
            result = await self.multiturn_results_collection.insert_one(doc)
            logger.debug(f"✅ Inserted multi-turn result: {result_data.get('conversation_id', 'unknown')[:8]}...")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"❌ Failed to insert multi-turn result: {e}")
            raise
    
    async def get_conversation_turns(
        self, 
        conversation_id: str
    ) -> List[Dict[str, Any]]:
        """Get all turns for a conversation"""
        try:
            cursor = self.conversations_collection.find(
                {"conversation_id": conversation_id}
            ).sort("turn_number", 1)
            
            results = await cursor.to_list(length=100)
            return results
        except Exception as e:
            logger.error(f"❌ Failed to get conversation turns: {e}")
            return []
    
    async def get_multiturn_result(
        self, 
        conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get complete multi-turn result"""
        try:
            result = await self.multiturn_results_collection.find_one(
                {"conversation_id": conversation_id}
            )
            return result
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
            query = {}
            if jailbroken_only:
                query["jailbreak_achieved"] = True
            
            cursor = self.multiturn_results_collection.find(query)\
                .sort("timestamp", -1)\
                .skip(skip)\
                .limit(limit)
            
            results = await cursor.to_list(length=limit)
            return results
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
            
            # Match stage
            match_stage = {
                "timestamp": {
                    "$gte": datetime.utcnow() - timedelta(days=days)
                }
            }
            
            if target_model:
                match_stage["target_model"] = target_model
            
            pipeline.append({"$match": match_stage})
            
            # Group and calculate
            pipeline.append({
                "$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "jailbroken": {
                        "$sum": {
                            "$cond": [
                                {"$eq": ["$jailbreak_achieved", True]},
                                1,
                                0
                            ]
                        }
                    }
                }
            })
            
            # Project ASR
            pipeline.append({
                "$project": {
                    "asr": {
                        "$cond": [
                            {"$eq": ["$total", 0]},
                            0,
                            {"$divide": ["$jailbroken", "$total"]}
                        ]
                    }
                }
            })
            
            cursor = self.multiturn_results_collection.aggregate(pipeline)
            results = await cursor.to_list(length=1)
            
            if results:
                return results[0].get("asr", 0.0)
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate multi-turn ASR: {e}")
            return 0.0
    
    async def get_strategy_effectiveness(
        self,
        days: int = 30
    ) -> Dict[str, float]:
        """Get strategy effectiveness across all multi-turn conversations"""
        try:
            pipeline = [
                {
                    "$match": {
                        "timestamp": {
                            "$gte": datetime.utcnow() - timedelta(days=days)
                        }
                    }
                },
                {
                    "$unwind": "$strategies_tried"
                },
                {
                    "$group": {
                        "_id": "$strategies_tried",
                        "total_uses": {"$sum": 1},
                        "successes": {
                            "$sum": {
                                "$cond": [
                                    {"$eq": ["$jailbreak_achieved", True]},
                                    1,
                                    0
                                ]
                            }
                        }
                    }
                },
                {
                    "$project": {
                        "strategy": "$_id",
                        "success_rate": {
                            "$cond": [
                                {"$eq": ["$total_uses", 0]},
                                0,
                                {"$divide": ["$successes", "$total_uses"]}
                            ]
                        }
                    }
                }
            ]
            
            cursor = self.multiturn_results_collection.aggregate(pipeline)
            results = await cursor.to_list(length=100)
            
            # Convert to dictionary
            effectiveness = {}
            for result in results:
                strategy = result.get("strategy", "unknown")
                success_rate = result.get("success_rate", 0.0)
                effectiveness[strategy] = success_rate
            
            return effectiveness
            
        except Exception as e:
            logger.error(f"❌ Failed to get strategy effectiveness: {e}")
            return {}


# Global database instance
db = Database()


async def get_database() -> Database:
    """Get database instance (for FastAPI dependency injection)"""
    return db
