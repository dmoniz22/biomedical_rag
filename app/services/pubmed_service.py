"""
PubMed API service for fetching biomedical literature
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import re
import json
from urllib.parse import quote, urlencode

from app.core.config import settings
from app.models.schemas import BulkIngestionRequest, IngestionJob

logger = logging.getLogger(__name__)


class PubMedService:
    """Service for interacting with PubMed/MEDLINE API"""
    
    def __init__(self):
        self.base_url = settings.PUBMED_BASE_URL
        self.api_key = settings.PUBMED_API_KEY
        self.session = None
    
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.REQUEST_TIMEOUT)
        )
        logger.info("PubMed service initialized")
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    def _build_search_params(self, query: str, max_results: int = 100, 
                           date_range: Optional[Dict[str, datetime]] = None) -> str:
        """Build PubMed search parameters"""
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "usehistory": "y"
        }
        
        # Add date range if specified
        if date_range:
            date_str = ""
            if date_range.get('start'):
                date_str += f"{date_range['start'].strftime('%Y/%m/%d')}[PDAT] : "
            if date_range.get('end'):
                date_str += f"{date_range['end'].strftime('%Y/%m/%d')}[PDAT]"
            
            if date_str:
                search_params["term"] = f"({query}) AND ({date_str})"
        
        # Add API key if available
        if self.api_key:
            search_params["api_key"] = self.api_key
        
        return urlencode(search_params)
    
    async def search_papers(self, query: str, max_results: int = 100,
                          date_range: Optional[Dict[str, datetime]] = None) -> List[str]:
        """Search for papers and return PMIDs"""
        try:
            search_url = f"{self.base_url}/esearch.fcgi"
            search_params = self._build_search_params(query, max_results, date_range)
            
            async with self.session.get(f"{search_url}?{search_params}") as response:
                if response.status == 200:
                    data = await response.json()
                    id_list = data.get("esearchresult", {}).get("idlist", [])
                    logger.info(f"Found {len(id_list)} papers for query: {query}")
                    return id_list
                else:
                    logger.error(f"PubMed search failed with status {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    async def fetch_paper_details(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch detailed information for papers by PMIDs"""
        try:
            if not pmids:
                return []
            
            # PubMed allows up to 200 PMIDs per request
            batch_size = 200
            papers = []
            
            for i in range(0, len(pmids), batch_size):
                batch = pmids[i:i + batch_size]
                batch_papers = await self._fetch_paper_batch(batch)
                papers.extend(batch_papers)
                
                # Add delay to respect rate limits
                await asyncio.sleep(0.1)
            
            logger.info(f"Fetched details for {len(papers)} papers")
            return papers
            
        except Exception as e:
            logger.error(f"Error fetching paper details: {e}")
            return []
    
    async def _fetch_paper_batch(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch details for a batch of PMIDs"""
        try:
            fetch_url = f"{self.base_url}/efetch.fcgi"
            params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "api_key": self.api_key
            }
            
            async with self.session.get(fetch_url, params=params) as response:
                if response.status == 200:
                    xml_content = await response.text()
                    return self._parse_pubmed_xml(xml_content)
                else:
                    logger.error(f"PubMed fetch failed with status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching paper batch: {e}")
            return []
    
    def _parse_pubmed_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse PubMed XML response and extract paper information"""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article in root.findall(".//PubmedArticle"):
                paper_data = self._parse_article_element(article)
                if paper_data:
                    papers.append(paper_data)
                    
        except ET.ParseError as e:
            logger.error(f"Error parsing PubMed XML: {e}")
        
        return papers
    
    def _parse_article_element(self, article) -> Optional[Dict[str, Any]]:
        """Parse individual PubMed article element"""
        try:
            # Extract PMID
            pmid_elem = article.find(".//MedlineCitation/PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None
            
            # Extract title
            title_elem = article.find(".//Article/ArticleTitle")
            title = title_elem.text if title_elem is not None else ""
            
            # Extract abstract
            abstract_text = ""
            abstract_elem = article.find(".//Article/Abstract")
            if abstract_elem is not None:
                abstract_parts = []
                for abstract_text_elem in abstract_elem.findall(".//AbstractText"):
                    if abstract_text_elem.text:
                        abstract_parts.append(abstract_text_elem.text)
                abstract_text = " ".join(abstract_parts)
            
            # Extract journal information
            journal_elem = article.find(".//Article/Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""
            
            # Extract publication date
            pub_date = None
            pub_date_elem = article.find(".//Article/Journal/JournalIssue/PubDate")
            if pub_date_elem is not None:
                year_elem = pub_date_elem.find("Year")
                month_elem = pub_date_elem.find("Month")
                day_elem = pub_date_elem.find("Day")
                
                year = year_elem.text if year_elem is not None else "1900"
                month = month_elem.text if month_elem is not None else "1"
                day = day_elem.text if day_elem is not None else "1"
                
                try:
                    pub_date = datetime(int(year), int(month), int(day))
                except ValueError:
                    pass
            
            # Extract DOI
            doi = None
            for id_elem in article.findall(".//ArticleIdList/ArticleId"):
                if id_elem.get("IdType") == "doi":
                    doi = id_elem.text
                    break
            
            # Extract MeSH terms
            mesh_terms = []
            for mesh_heading in article.findall(".//MeshHeadingList/MeshHeading"):
                descriptor_elem = mesh_heading.find("DescriptorName")
                if descriptor_elem is not None and descriptor_elem.text:
                    mesh_terms.append(descriptor_elem.text)
            
            # Extract authors
            authors = []
            for author in article.findall(".//AuthorList/Author"):
                last_name = author.find("LastName")
                first_name = author.find("ForeName")
                
                author_name = []
                if last_name is not None:
                    author_name.append(last_name.text)
                if first_name is not None:
                    author_name.append(first_name.text)
                
                if author_name:
                    authors.append(" ".join(author_name))
            
            # Extract keywords
            keywords = []
            for keyword in article.findall(".//KeywordList/Keyword"):
                if keyword.text:
                    keywords.append(keyword.text)
            
            # Extract publication type
            pub_type = None
            pub_type_elem = article.find(".//PublicationTypeList/PublicationType")
            if pub_type_elem is not None and pub_type_elem.text:
                pub_type = pub_type_elem.text
            
            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract_text,
                "journal": journal,
                "publication_date": pub_date,
                "doi": doi,
                "mesh_terms": mesh_terms,
                "keywords": keywords,
                "authors": authors,
                "publication_type": pub_type,
                "source_database": "pubmed"
            }
            
        except Exception as e:
            logger.error(f"Error parsing article element: {e}")
            return None
    
    async def bulk_search_and_fetch(self, ingestion_request: BulkIngestionRequest) -> List[Dict[str, Any]]:
        """Perform bulk search and fetch for multiple subject areas"""
        all_papers = []
        
        for subject_area in ingestion_request.subject_areas:
            # Build search query for subject area
            search_query = self._build_subject_area_query(subject_area)
            
            # Add date range filter if specified
            date_range = None
            if ingestion_request.date_range_start or ingestion_request.date_range_end:
                date_range = {
                    "start": ingestion_request.date_range_start,
                    "end": ingestion_request.date_range_end
                }
            
            # Search for papers
            pmids = await self.search_papers(
                query=search_query,
                max_results=ingestion_request.max_documents or 1000,
                date_range=date_range
            )
            
            if pmids:
                # Fetch paper details
                papers = await self.fetch_paper_details(pmids)
                
                # Add subject area to each paper
                for paper in papers:
                    paper["subject_areas"] = [subject_area]
                    paper["bulk_ingestion"] = True
                
                all_papers.extend(papers)
                
                logger.info(f"Fetched {len(papers)} papers for subject area: {subject_area}")
            
            # Add delay between subject areas
            await asyncio.sleep(1)
        
        return all_papers
    
    def _build_subject_area_query(self, subject_area: str) -> str:
        """Build PubMed search query for a subject area"""
        # Map subject areas to MeSH terms and keywords
        subject_queries = {
            "cardiology": "cardiology[MeSH Terms] OR cardiovascular[tiab] OR heart disease[tiab]",
            "oncology": "oncology[MeSH Terms] OR cancer[tiab] OR neoplasms[tiab]",
            "neurology": "neurology[MeSH Terms] OR neurological[tiab] OR brain disease[tiab]",
            "immunology": "immunology[MeSH Terms] OR immune[tiab] OR autoimmune[tiab]",
            "endocrinology": "endocrinology[MeSH Terms] OR diabetes[tiab] OR hormone[tiab]",
            "gastroenterology": "gastroenterology[MeSH Terms] OR digestive[tiab] OR gastrointestinal[tiab]",
            "nephrology": "nephrology[MeSH Terms] OR kidney[tiab] OR renal[tiab]",
            "pulmonology": "pulmonology[MeSH Terms] OR lung[tiab] OR pulmonary[tiab]",
            "rheumatology": "rheumatology[MeSH Terms] OR arthritis[tiab] OR autoimmune[tiab]",
            "dermatology": "dermatology[MeSH Terms] OR skin[tiab] OR dermatology[tiab]",
            "ophthalmology": "ophthalmology[MeSH Terms] OR eye[tiab] OR vision[tiab]",
            "psychiatry": "psychiatry[MeSH Terms] OR mental health[tiab] OR depression[tiab]",
            "pediatrics": "pediatrics[MeSH Terms] OR child[tiab] OR pediatric[tiab]",
            "geriatrics": "geriatrics[MeSH Terms] OR elderly[tiab] OR aging[tiab]"
        }
        
        return subject_queries.get(subject_area.lower(), f'"{subject_area}"[tiab]')
    
    async def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about available data"""
        try:
            # This would typically query PubMed for database statistics
            # For now, return basic info
            return {
                "total_pubmed_articles": "35+ million",
                "updated_daily": True,
                "api_version": "2.0",
                "rate_limits": "3 requests per second without key, 10 with key"
            }
        except Exception as e:
            logger.error(f"Error getting search stats: {e}")
            return {}


# Global PubMed service instance
pubmed_service = PubMedService()