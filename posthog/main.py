import os
import requests
from dotenv import load_dotenv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

load_dotenv()

class PostHogAPI:
    def __init__(self):
        self.personal_api_key = os.getenv('POSTHOG_PERSONAL_API_KEY')
        self.project_id = os.getenv('POSTHOG_PROJECT_ID')
        self.host = os.getenv('POSTHOG_HOST', 'https://us.posthog.com')
        self.base_url = f'{self.host}/api'
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {self.personal_api_key}'})

    def _parse_response(self, response: requests.Response) -> Any:
        """Parse response that might be JSON or NDJSON"""
        text = response.text.strip()
        if not text:
            return None
            
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                lines = text.strip().split('\n')
                if len(lines) == 1:
                    raise json.JSONDecodeError("Single line JSON parse failed", text, 0)
                
                parsed_objects = []
                for line in lines:
                    if line.strip():
                        parsed_objects.append(json.loads(line.strip()))
                
                return parsed_objects if len(parsed_objects) > 1 else parsed_objects[0]
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse response: {e}")
                print(f"Response text: {text[:500]}...")
                raise

    def list_sources(self, recording_id: str, prefer_v2=True) -> Dict[str, Any]:
        params = {'blob_v2': 'true'} if prefer_v2 else {}
        url = f"{self.base_url}/environments/{self.project_id}/session_recordings/{recording_id}/snapshots"
        r = self.session.get(url, params=params)
        r.raise_for_status()
        return self._parse_response(r)

    def fetch_blob_v2_single(self, recording_id: str, blob_key: str) -> Dict[str, Any]:
        params = {'source': 'blob_v2', 'blob_key': str(blob_key)}
        url = f"{self.base_url}/environments/{self.project_id}/session_recordings/{recording_id}/snapshots"
        r = self.session.get(url, params=params)
        r.raise_for_status()
        return self._parse_response(r)

    def get_recording_events(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get events with simple HogQL query"""
        headers = {'Content-Type': 'application/json'}
        url = f"{self.base_url}/projects/{self.project_id}/query"
        
        # Use proper HogQL syntax
        query = """
            SELECT 
                event,
                timestamp,
                properties
            FROM events 
            WHERE properties['$session_id'] = {session_id}
            ORDER BY timestamp ASC 
            LIMIT 2000
        """
        
        payload = {
            "query": {
                "kind": "HogQLQuery", 
                "query": query,
                "params": {"session_id": session_id}
            }
        }
        
        try:
            response = self.session.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return self._parse_response(response)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching recording events: {e}")
            return None

    @staticmethod
    def _has_snapshot_payload(data: Any) -> bool:
        if data is None:
            return False
        if isinstance(data, list):
            return len(data) > 0 and any(isinstance(item, dict) for item in data)
        if isinstance(data, dict):
            return any(isinstance(v, list) and len(v) > 0 for v in data.values())
        return False

    def save_recording(self, recording_id: str, output_dir="recordings") -> Optional[str]:
        out = Path(output_dir) / recording_id
        out.mkdir(parents=True, exist_ok=True)

        try:
            sources_resp = self.list_sources(recording_id, prefer_v2=True)
        except Exception as e:
            print(f"Error getting sources: {e}")
            return None
            
        (out / 'sources.json').write_text(json.dumps(sources_resp, indent=2))

        sources = sources_resp.get('sources', []) if isinstance(sources_resp, dict) else []
        if not sources:
            print("No sources returned. If the session is fresh, wait and retry.")
            return None

        # Get events data
        events = self.get_recording_events(recording_id)
        if events:
            (out / 'events.json').write_text(json.dumps(events, indent=2))
            print("Saved events.json")

        # Handle blob fetching
        blob_v2_sources = [s for s in sources if s.get('source') == 'blob_v2']

        if blob_v2_sources:
            print(f"Found {len(blob_v2_sources)} blob_v2 source(s). Fetching individually...")
            
            success_count = 0
            for source in blob_v2_sources:
                blob_key = source['blob_key']
                try:
                    payload = self.fetch_blob_v2_single(recording_id, blob_key)
                    if self._has_snapshot_payload(payload):
                        fname = out / f"blob_v2_{blob_key}.json"
                        fname.write_text(json.dumps(payload, indent=2))
                        print(f"Saved {fname.name}")
                        success_count += 1
                    else:
                        print(f"No data in blob {blob_key}")
                except Exception as e:
                    print(f"Error processing blob {blob_key}: {e}")
                    
            if success_count == 0:
                print("No blob_v2 data retrieved. Recording may be too fresh.")

        return str(out)

class SessionAnalyzer:
    def __init__(self, recording_dir: str):
        self.recording_dir = Path(recording_dir)
        
    def load_events(self) -> List[Dict[str, Any]]:
        """Load events from PostHog results array format"""
        events_file = self.recording_dir / 'events.json'
        if not events_file.exists():
            return []
            
        with open(events_file) as f:
            events_data = json.load(f)
            
        events = []
        # Handle the actual format: results is a list of arrays where each array is [event, timestamp, properties]
        for row in events_data.get('results', []):
            if len(row) >= 3:  # Need at least event, timestamp, properties
                try:
                    # Parse properties JSON string
                    properties = json.loads(row[2]) if isinstance(row[2], str) else row[2]
                    
                    events.append({
                        'event': row[0],
                        'timestamp': row[1],
                        'properties': properties,
                        'url': properties.get('$current_url'),
                        'pathname': properties.get('$pathname'),
                        'element_text': properties.get('$el_text'),
                        'event_type': properties.get('$event_type'),
                        'recording_status': properties.get('$recording_status'),
                        'distinct_id': properties.get('distinct_id')
                    })
                except (json.JSONDecodeError, IndexError, KeyError) as e:
                    print(f"Error parsing event row {row}: {e}")
                    continue
        
        # Sort by timestamp
        events.sort(key=lambda e: e['timestamp'])
        return events
    
    def load_rrweb_snapshots(self) -> List[Dict[str, Any]]:
        """Load and parse rrweb snapshots"""
        snapshots = []
        
        blob_files = list(self.recording_dir.glob("blob_v2_*.json"))
        blob_files.sort()
        
        print(f"Found {len(blob_files)} blob files to process")
        
        for blob_file in blob_files:
            try:
                with open(blob_file) as f:
                    blob_data = json.load(f)
                
                print(f"Processing {blob_file.name}, type: {type(blob_data)}")
                
                if isinstance(blob_data, list):
                    for item in blob_data:
                        if isinstance(item, dict) and 'timestamp' in item:
                            snapshots.append(item)
                elif isinstance(blob_data, dict):
                    if 'timestamp' in blob_data:
                        snapshots.append(blob_data)
                            
            except Exception as e:
                print(f"Error loading {blob_file}: {e}")
        
        valid_snapshots = [s for s in snapshots if isinstance(s, dict) and 'timestamp' in s]
        
        try:
            valid_snapshots.sort(key=lambda s: s.get('timestamp', 0))
        except Exception as e:
            print(f"Error sorting snapshots: {e}")
        
        print(f"Total valid snapshots loaded: {len(valid_snapshots)}")
        return valid_snapshots
    
    def extract_console_logs(self, snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract console logs and errors from rrweb snapshots"""
        console_logs = []
        
        for snapshot in snapshots:
            try:
                # RRWeb console logs are typically in type 5 (Custom) snapshots
                if snapshot.get('type') == 5:  # Custom snapshot type
                    data = snapshot.get('data', {})
                    
                    # Check if this is a console log
                    if data.get('tag') == 'rrweb/console':
                        payload = data.get('payload', {})
                        console_logs.append({
                            'timestamp': snapshot.get('timestamp'),
                            'level': payload.get('level', 'log'),
                            'message': payload.get('payload', []),
                            'trace': payload.get('trace', [])
                        })
                
                # Also check incremental snapshots (type 3) for console data
                elif snapshot.get('type') == 3:  # IncrementalSnapshot
                    data = snapshot.get('data', {})
                    
                    # Console logs in incremental snapshots
                    if data.get('source') == 8:  # Console source
                        console_logs.append({
                            'timestamp': snapshot.get('timestamp'),
                            'level': data.get('level', 'log'),
                            'message': data.get('payload', []),
                            'trace': data.get('trace', [])
                        })
                    
                    # Plugin snapshots that might contain console data
                    elif data.get('plugin') == 'rrweb/console@1':
                        payload = data.get('payload', {})
                        console_logs.append({
                            'timestamp': snapshot.get('timestamp'),
                            'level': payload.get('level', 'log'),
                            'message': payload.get('payload', []),
                            'trace': payload.get('trace', [])
                        })
                        
            except Exception as e:
                # Skip problematic snapshots
                continue
        
        return console_logs
    
    def extract_network_errors(self, snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract network errors from rrweb snapshots"""
        network_errors = []
        
        for snapshot in snapshots:
            try:
                if snapshot.get('type') == 3:  # IncrementalSnapshot
                    data = snapshot.get('data', {})
                    
                    # Network source
                    if data.get('source') == 7:  # Network source
                        if data.get('isError') or (data.get('status') and data.get('status') >= 400):
                            network_errors.append({
                                'timestamp': snapshot.get('timestamp'),
                                'url': data.get('url'),
                                'method': data.get('method'),
                                'status': data.get('status'),
                                'error': data.get('error'),
                                'response': data.get('response')
                            })
                            
            except Exception as e:
                continue
        
        return network_errors
    
    def analyze_user_journey(self) -> Dict[str, Any]:
        """Analyze the user journey with console logs and network errors"""
        events = self.load_events()
        snapshots = self.load_rrweb_snapshots()
        console_logs = self.extract_console_logs(snapshots)
        network_errors = self.extract_network_errors(snapshots)
        
        print(f"Extracted {len(console_logs)} console logs and {len(network_errors)} network errors")
        
        if not events:
            return {
                'total_events': 0,
                'total_snapshots': len(snapshots),
                'session_duration': 0.0,
                'console_logs': console_logs,
                'network_errors': network_errors,
                'error': 'No events found'
            }
        
        # Calculate session duration
        try:
            first_event = datetime.fromisoformat(events[0]['timestamp'].replace('Z', '+00:00'))
            last_event = datetime.fromisoformat(events[-1]['timestamp'].replace('Z', '+00:00'))
            session_duration = (last_event - first_event).total_seconds()
        except Exception as e:
            print(f"Error calculating session duration: {e}")
            session_duration = 0.0
        
        # Extract page transitions
        page_transitions = []
        current_url = None
        
        for event in events:
            new_url = event['url']
            event_type = event['event']
            
            # Track page transitions
            if new_url and new_url != current_url:
                page_transitions.append({
                    'timestamp': event['timestamp'],
                    'event_type': event_type,
                    'from': current_url,
                    'to': new_url,
                    'pathname': event['pathname']
                })
                current_url = new_url
        
        # Track user actions
        user_actions = []
        for event in events:
            if event['event'] == '$autocapture' and event['element_text']:
                user_actions.append({
                    'timestamp': event['timestamp'],
                    'type': event['event_type'] or 'unknown',
                    'element': event['element_text'],
                    'url': event['url']
                })
        
        # Check for issues
        recording_issues = []
        for event in events:
            if event['recording_status'] == 'disabled':
                recording_issues.append({
                    'timestamp': event['timestamp'],
                    'issue': 'Recording was disabled'
                })
        
        # Find actual errors (not normal events)
        javascript_errors = []
        for event in events:
            if event['event'] in ['$exception', '$error'] or 'error' in event['event'].lower():
                javascript_errors.append({
                    'timestamp': event['timestamp'],
                    'event': event['event'],
                    'details': event['properties']
                })
        
        # Filter console logs for errors and warnings
        console_errors = [log for log in console_logs if log['level'] in ['error', 'warn']]
        
        return {
            'total_events': len(events),
            'total_snapshots': len(snapshots),
            'session_duration': session_duration,
            'page_transitions': page_transitions,
            'user_actions': user_actions,
            'recording_issues': recording_issues,
            'javascript_errors': javascript_errors,
            'console_logs': console_logs,
            'console_errors': console_errors,
            'network_errors': network_errors,
            'distinct_ids': list(set(e['distinct_id'] for e in events if e['distinct_id']))
        }
    
    def generate_bug_report(self) -> str:
        """Generate a comprehensive bug analysis report with console logs"""
        analysis = self.analyze_user_journey()
        
        if 'error' in analysis:
            return f"## Error\n{analysis['error']}"
        
        # Keep session_duration as number for calculations
        session_duration = analysis['session_duration'] or 0.0
        duration_str = f"{session_duration:.1f}"
        duration_minutes = session_duration / 60
        
        report = f"""# Session Analysis Report

## Overview
- **Total Events**: {analysis['total_events']}
- **Total Snapshots**: {analysis['total_snapshots']}
- **Session Duration**: {duration_str} seconds ({duration_minutes:.1f} minutes)
- **Page Transitions**: {len(analysis['page_transitions'])}
- **User Actions**: {len(analysis['user_actions'])}
- **JavaScript Errors**: {len(analysis['javascript_errors'])}
- **Console Logs**: {len(analysis.get('console_logs', []))}
- **Console Errors/Warnings**: {len(analysis.get('console_errors', []))}
- **Network Errors**: {len(analysis.get('network_errors', []))}
- **Recording Issues**: {len(analysis['recording_issues'])}
- **Distinct Users**: {len(analysis['distinct_ids'])}

## User Journey
"""
        
        if analysis['page_transitions']:
            for i, transition in enumerate(analysis['page_transitions']):
                from_url = transition.get('from') or 'START'
                to_url = transition.get('to') or 'UNKNOWN'
                
                try:
                    from_path = from_url.split('/')[-1] if from_url != 'START' else 'START'
                    to_path = to_url.split('/')[-1] if to_url != 'UNKNOWN' else 'UNKNOWN'
                except (AttributeError, IndexError):
                    from_path = str(from_url)
                    to_path = str(to_url)
                
                report += f"{i+1}. **{transition['timestamp']}**: {from_path} → {to_path} (via {transition['event_type']})\n"
                report += f"   - Full URL: {to_url}\n"
        else:
            report += "No page transitions detected.\n"
        
        # Console Errors Section
        console_errors = analysis.get('console_errors', [])
        if console_errors:
            report += f"\n## 🔴 Console Errors & Warnings\n"
            for log in console_errors:
                level_emoji = "🔴" if log['level'] == 'error' else "🟡"
                message = log['message']
                if isinstance(message, list):
                    message = ' '.join(str(m) for m in message)
                report += f"{level_emoji} **{log['timestamp']}** [{log['level'].upper()}]: {message}\n"
        
        # All Console Logs Section (if you want to see everything)
        console_logs = analysis.get('console_logs', [])
        if console_logs:
            report += f"\n## 📝 All Console Logs\n"
            for log in console_logs[-10:]:  # Show last 10 logs
                level_emoji = {"error": "🔴", "warn": "🟡", "info": "ℹ️", "log": "📝", "debug": "🐛"}.get(log['level'], "📝")
                message = log['message']
                if isinstance(message, list):
                    message = ' '.join(str(m) for m in message)
                report += f"{level_emoji} **{log['timestamp']}** [{log['level'].upper()}]: {message}\n"
        
        # Network Errors Section
        network_errors = analysis.get('network_errors', [])
        if network_errors:
            report += f"\n## 🌐 Network Errors\n"
            for error in network_errors:
                report += f"🔴 **{error['timestamp']}**: {error.get('method', 'GET')} {error.get('url', 'unknown')} - Status: {error.get('status', 'unknown')}\n"
        
        if analysis['user_actions']:
            report += f"\n## User Actions\n"
            for action in analysis['user_actions']:
                element_text = action.get('element') or 'unknown element'
                action_type = action.get('type') or 'unknown'
                report += f"- **{action['timestamp']}**: {action_type} on '{element_text}'\n"
        
        if analysis['recording_issues']:
            report += "\n## ⚠️  Recording Issues\n"
            for issue in analysis['recording_issues']:
                issue_desc = issue.get('issue', 'Unknown issue')
                report += f"- **{issue['timestamp']}**: {issue_desc}\n"
        
        if analysis['javascript_errors']:
            report += "\n## 🐛 JavaScript Errors (from Events)\n"
            for error in analysis['javascript_errors']:
                error_event = error.get('event', 'Unknown error')
                report += f"- **{error['timestamp']}**: {error_event}\n"
        
        # Analysis insights
        report += "\n## 🔍 Analysis Insights\n"
        
        distinct_ids = analysis.get('distinct_ids', [])
        if len(distinct_ids) > 1:
            report += f"- ⚠️  Multiple users detected: {distinct_ids}\n"
        
        if analysis.get('recording_issues'):
            report += "- ⚠️  Recording was disabled during parts of the session\n"
        
        total_snapshots = analysis.get('total_snapshots', 0)
        if total_snapshots == 0:
            report += "- ⚠️  No visual recording data found\n"
        elif total_snapshots < 10:
            report += f"- ⚠️  Limited visual recording data ({total_snapshots} snapshots)\n"
        
        if console_errors:
            report += f"- 🔴 {len(console_errors)} console errors/warnings detected\n"
        
        if network_errors:
            report += f"- 🌐 {len(network_errors)} network errors detected\n"
        
        # Recommendations
        report += "\n## 💡 Recommendations\n"
        
        javascript_errors = analysis.get('javascript_errors', [])
        recording_issues = analysis.get('recording_issues', [])
        
        if not javascript_errors and not recording_issues and not console_errors and not network_errors:
            report += "- ✅ Session appears healthy - no obvious errors detected\n"
        
        if console_errors:
            report += "- 🔧 Investigate console errors and warnings for potential issues\n"
        
        if network_errors:
            report += "- 🌐 Check network errors for API failures or connectivity issues\n"
        
        if recording_issues:
            report += "- 🔧 Investigate why recording was disabled during the session\n"
        
        page_transitions = analysis.get('page_transitions', [])
        if len(page_transitions) > 5:
            report += "- 📊 High navigation activity - consider if this matches expected user flow\n"
        
        return report

if __name__ == "__main__":
    # Use existing recording directory
    recording_dir = "recordings/01985903-5f64-785f-878a-438b13625f6d"
    
    if Path(recording_dir).exists():
        print(f"Analyzing existing recording: {recording_dir}")
        
        analyzer = SessionAnalyzer(recording_dir)
        report = analyzer.generate_bug_report()
        
        report_file = Path(recording_dir) / 'complete_analysis_report.md'
        report_file.write_text(report)
        print(f"Analysis report saved to: {report_file}")
        print("\n" + "="*50)
        print(report)
    else:
        print(f"Recording directory not found: {recording_dir}")
