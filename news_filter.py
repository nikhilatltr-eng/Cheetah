import logging
import datetime
import urllib.request
import xml.etree.ElementTree as ET
import os
import json

logger = logging.getLogger(__name__)

class NewsFilter:
    def __init__(self, blackout_minutes: int = 15, api_key: str = None, 
                 feed_url: str = "https://nfs.faireconomy.media/lw_cal_dec.xml"):
        """
        Polls the FairEconomy weekly economic calendar feed or Finnhub API (if api_key is configured),
        filters for high-impact USD events, and checks for entry blackout periods.
        """
        self.blackout_minutes = blackout_minutes
        self.api_key = api_key
        self.feed_url = feed_url
        self.high_impact_events = []  # List of datetime (UTC) objects of high-impact USD events

    def fetch_calendar_events(self) -> list:
        """
        Downloads and parses the economic calendar.
        Attempts to use Finnhub API if api_key is provided; falls back to FairEconomy XML.
        """
        if self.api_key:
            try:
                logger.info("NewsFilter: Fetching calendar events from Finnhub API...")
                url = f"https://finnhub.io/api/v1/calendar/economic?token={self.api_key}"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                
                with urllib.request.urlopen(req, timeout=10) as response:
                    res_json = json.loads(response.read())
                    
                parsed_times = []
                events = res_json.get("economicCalendar", [])
                for event in events:
                    country = event.get("country", "")
                    impact = event.get("impact", "")
                    time_str = event.get("time", "")  # Expected format: "YYYY-MM-DD HH:MM:SS" (UTC)
                    
                    if country == "US" and impact == "high":
                        try:
                            # Replace space with T for ISO format parsing
                            dt = datetime.datetime.fromisoformat(time_str.replace(" ", "T"))
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=datetime.timezone.utc)
                            parsed_times.append(dt)
                        except Exception as e:
                            logger.warning(f"NewsFilter: Error parsing Finnhub timestamp '{time_str}': {e}")
                            
                if parsed_times:
                    self.high_impact_events = parsed_times
                    logger.info(f"NewsFilter: Successfully registered {len(self.high_impact_events)} events from Finnhub.")
                    return self.high_impact_events
                    
            except Exception as e:
                logger.error(f"NewsFilter: Finnhub API fetch failed: {e}. Falling back to FairEconomy weekly feed...")

        # FairEconomy XML Feed Fallback
        logger.info(f"NewsFilter: Fetching weekly calendar events from FairEconomy XML: {self.feed_url}...")
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(self.feed_url, headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()
                
            root = ET.fromstring(xml_data)
            parsed_times = []
            
            for event in root.findall("event"):
                country = event.find("country").text
                impact = event.find("impact").text
                
                # Check for High-Impact USD events (critical for gold trading)
                if country == "USD" and impact == "High":
                    date_val = event.find("date").text  # Format: MM-DD-YYYY
                    time_val = event.find("time").text  # Format: hh:mm AM/PM
                    title = event.find("title").text
                    
                    try:
                        # Combine and parse (assuming EST/EDT for the weekly feed)
                        dt_str = f"{date_val} {time_val}"
                        # Parse Eastern time
                        dt_est = datetime.datetime.strptime(dt_str, "%m-%d-%Y %I:%M %p")
                        
                        # Convert EST to UTC (Eastern is roughly UTC-5)
                        dt_utc = dt_est + datetime.timedelta(hours=5)
                        dt_utc = dt_utc.replace(tzinfo=datetime.timezone.utc)
                        
                        parsed_times.append(dt_utc)
                        logger.debug(f"NewsFilter: Registered high-impact event '{title}' at {dt_utc}")
                    except Exception as e:
                        logger.warning(f"NewsFilter: Error parsing event timestamp '{date_val} {time_val}': {e}")
                        
            self.high_impact_events = parsed_times
            logger.info(f"NewsFilter: Successfully registered {len(self.high_impact_events)} high-impact USD events.")
            return self.high_impact_events
            
        except Exception as e:
            logger.error(f"NewsFilter: Failed to fetch economic calendar XML feed: {e}.")
            return []

    def is_blackout_active(self, current_time: datetime.datetime) -> bool:
        """
        Returns True if current_time falls within the blackout window of any high-impact event.
        """
        # Ensure current_time is timezone-aware
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=datetime.timezone.utc)
            
        for event_time in self.high_impact_events:
            # Ensure event_time is timezone-aware
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=datetime.timezone.utc)
                
            time_diff = abs((current_time - event_time).total_seconds()) / 60.0
            if time_diff <= self.blackout_minutes:
                logger.warning(
                    f"NewsFilter Blackout Active: Proximity of {time_diff:.1f} minutes to news release "
                    f"at {event_time.isoformat()} (Limit: {self.blackout_minutes} mins)."
                )
                return True
        return False

    def add_mock_event(self, event_time: datetime.datetime):
        """Helper to inject custom events during unit testing."""
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=datetime.timezone.utc)
        self.high_impact_events.append(event_time)
        logger.info(f"NewsFilter: Added mock news event at {event_time.isoformat()}")
