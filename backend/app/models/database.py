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
            
            logger.info("✅ Database indexes created")
            
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


# Global database instance
db = Database()


async def get_database() -> Database:
    """Get database instance (for FastAPI dependency injection)"""
    return db
