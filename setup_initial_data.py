#!/usr/bin/env python3
"""
Initial data seeding script for biomedical RAG system
This script sets up the system with initial data and configuration
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.database import db_manager
from app.models.database import Paper, Author, SubjectArea
from app.core.config import settings
from app.services.bulk_ingestion_service import bulk_ingestion_service
from app.models.schemas import BulkIngestionRequest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def setup_database():
    """Initialize database and create all tables"""
    try:
        logger.info("Setting up database...")
        await db_manager.create_tables()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to setup database: {e}")
        raise


async def create_subject_areas():
    """Create initial subject areas for classification"""
    try:
        logger.info("Creating subject areas...")
        
        subject_areas_data = [
            {
                "name": "cardiology",
                "display_name": "Cardiology",
                "description": "Heart and cardiovascular system diseases",
                "mesh_keywords": ["Heart Diseases", "Cardiovascular System", "Myocardial Infarction"],
                "keyword_patterns": [r"heart", r"cardio", r"cardiac", r"hypertension", r"blood pressure"],
                "priority": 1
            },
            {
                "name": "oncology",
                "display_name": "Oncology",
                "description": "Cancer and tumor research",
                "mesh_keywords": ["Neoplasms", "Cancer", "Tumor", "Oncology"],
                "keyword_patterns": [r"cancer", r"tumor", r"oncology", r"chemotherapy", r"malignancy"],
                "priority": 1
            },
            {
                "name": "neurology",
                "display_name": "Neurology",
                "description": "Nervous system and brain disorders",
                "mesh_keywords": ["Nervous System", "Brain Diseases", "Neurology"],
                "keyword_patterns": [r"brain", r"neurology", r"alzheimer", r"parkinson", r"epilepsy"],
                "priority": 1
            },
            {
                "name": "endocrinology",
                "display_name": "Endocrinology",
                "description": "Hormones and metabolic disorders",
                "mesh_keywords": ["Endocrine System", "Diabetes Mellitus", "Hormones"],
                "keyword_patterns": [r"diabetes", r"hormone", r"endocrine", r"insulin", r"thyroid"],
                "priority": 1
            }
        ]
        
        async with db_manager.get_session() as session:
            for area_data in subject_areas_data:
                # Check if subject area already exists
                existing = await session.execute(
                    "SELECT id FROM subject_areas WHERE name = :name",
                    {"name": area_data["name"]}
                )
                
                if not existing.fetchone():
                    subject_area = SubjectArea(
                        name=area_data["name"],
                        display_name=area_data["display_name"],
                        description=area_data["description"],
                        mesh_keywords=area_data["mesh_keywords"],
                        keyword_patterns=area_data["keyword_patterns"],
                        priority=area_data["priority"],
                        is_active=True
                    )
                    session.add(subject_area)
                    logger.info(f"Created subject area: {area_data['display_name']}")
            
            await session.commit()
            logger.info("Subject areas setup completed")
            
    except Exception as e:
        logger.error(f"Failed to create subject areas: {e}")
        raise


async def main():
    """Main setup function"""
    try:
        logger.info("=" * 60)
        logger.info("Biomedical RAG System - Initial Data Seeding")
        logger.info("=" * 60)
        
        # Step 1: Setup database
        await setup_database()
        
        # Step 2: Create subject areas
        await create_subject_areas()
        
        logger.info("=" * 60)
        logger.info("Initial data seeding completed successfully!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Start the application: python -m uvicorn app.main:app --reload")
        logger.info("2. Access API documentation: http://localhost:8000/docs")
        logger.info("3. Test search endpoint: POST /api/v1/search")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())