import json
from locale import locale_encoding_alias
import os
import logging
import re
from typing import Dict, Any, List

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from backend.src.graph.state import VideoAuditState, ComplianceIssue
from backend.src.services.video_indexer import VideoIndexerService


logger = logging.getLogger("Brand-guardian")

def index_video_node(state: VideoAuditState) -> Dict[str, Any]:
    """
    Index video content using Video Indexer service.
    
    Args:
        state: Current graph state containing video data
        
    Returns:
        Updated state with indexed video content
    """
    video_url = state.get("video_url")
    
    video_id_input = state.get("video_id","vid_demo")
    
    logger.info(f"Indexing video: {video_url}")
    logger.info(f"Video ID: {video_id_input}")
    
    local_filename = "temp_audit_video.mp4"
    
    try:
        vi_service = VideoIndexerService()
        if "youtube.com" in video_url or "youtu.be" in video_url:
            local_path = vi_service.download_youtube_video(video_url, output_path=local_filename)
        else:
            raise Exception("please provide a valid youtube url")
        
        azure_video_id = vi_service.upload_video(local_path, video_name = video_id_input)
        logger.info(f"Video uploaded successfully with ID: {azure_video_id}")
        
        if os.path.exists(local_path):
            os.remove(local_path)
            logger.info(f"Cleaned up temporary file: {local_path}")
        
        raw_insights = vi_service.wait_for_processing(azure_video_id)
        clean_data = vi_service.extract_data(raw_insights)
        logger.info(f"Data extraction completed")
        
        return clean_data
    except Exception as e:
        logger.error(f"Error indexing video: {str(e)}")
        return {
            "error": str(e),
            "final_result": "failed",
            "transcript": "",
            "ocr_text": ""
        }
        