#!/usr/bin/env python3
"""
YouTube Downloader CLI
A simple command-line tool to download YouTube videos with audio, extract MP3, and generate transcriptions.
Now supports playlist downloads!
"""

import os
import re
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

def find_pirate_station_folder():
    """Find USB drive with pirate_station folder or prompt user to create one."""
    system = platform.system()
    possible_drives = []
    
    if system == "Windows":
        # Check all drive letters from D to Z
        for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
            drive_path = f"{letter}:\\"
            if os.path.exists(drive_path):
                # Try to determine if it's a removable drive
                try:
                    # Check if we can write to it (basic test for USB drives)
                    test_file = os.path.join(drive_path, "test_write_access.tmp")
                    with open(test_file, 'w') as f:
                        f.write("test")
                    os.remove(test_file)
                    possible_drives.append(drive_path)
                except (PermissionError, OSError):
                    # If we can't write, it might be a CD/DVD or protected drive
                    # Still add it as a possible option
                    possible_drives.append(drive_path)
    
    elif system == "Darwin":  # macOS
        # Check /Volumes for mounted drives
        volumes_path = "/Volumes"
        if os.path.exists(volumes_path):
            for volume in os.listdir(volumes_path):
                volume_path = os.path.join(volumes_path, volume)
                if os.path.isdir(volume_path) and volume != "Macintosh HD":
                    possible_drives.append(volume_path)
    
    elif system == "Linux":
        # Check common mount points
        mount_points = ["/media", "/mnt", "/run/media"]
        for mount_point in mount_points:
            if os.path.exists(mount_point):
                # Look for user directories in /media and /run/media
                if mount_point in ["/media", "/run/media"]:
                    try:
                        for user_dir in os.listdir(mount_point):
                            user_path = os.path.join(mount_point, user_dir)
                            if os.path.isdir(user_path):
                                for drive in os.listdir(user_path):
                                    drive_path = os.path.join(user_path, drive)
                                    if os.path.isdir(drive_path):
                                        possible_drives.append(drive_path)
                    except PermissionError:
                        continue
                else:
                    # Direct mounts in /mnt
                    try:
                        for drive in os.listdir(mount_point):
                            drive_path = os.path.join(mount_point, drive)
                            if os.path.isdir(drive_path):
                                possible_drives.append(drive_path)
                    except PermissionError:
                        continue
    
    # Look for pirate_station folder in possible drives
    print("🔍 Searching for USB drives with 'pirate_station' folder...")
    
    for drive in possible_drives:
        pirate_station_path = os.path.join(drive, "pirate_station")
        if os.path.exists(pirate_station_path) and os.path.isdir(pirate_station_path):
            print(f"✅ Found pirate_station folder at: {pirate_station_path}")
            return pirate_station_path
    
    # If not found, prompt user
    print("\n❌ No USB drive with 'pirate_station' folder found!")
    
    if possible_drives:
        print("\nDetected drives:")
        for i, drive in enumerate(possible_drives, 1):
            # Show drive info
            try:
                free_space = get_drive_free_space(drive)
                print(f"   • {drive} ({free_space})")
            except:
                print(f"   • {drive}")
    else:
        print("\nNo external drives detected")
    
    print("\nWhat would you like to do?")
    print("   1. Insert a USB drive and try again")
    print("   2. Create 'pirate_station' folder on one of the detected drives")
    print("   3. Use current directory instead")
    print("   4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            input("Insert your USB drive and press Enter to continue...")
            return find_pirate_station_folder()  # Recursive call to search again
        
        elif choice == "2":
            if not possible_drives:
                print("❌ No drives available to create folder on")
                continue
            
            print("\nSelect a drive to create 'pirate_station' folder:")
            for i, drive in enumerate(possible_drives, 1):
                try:
                    free_space = get_drive_free_space(drive)
                    print(f"   {i}. {drive} ({free_space})")
                except:
                    print(f"   {i}. {drive}")
            
            try:
                drive_choice = int(input("\nEnter drive number: ")) - 1
                if 0 <= drive_choice < len(possible_drives):
                    selected_drive = possible_drives[drive_choice]
                    pirate_station_path = os.path.join(selected_drive, "pirate_station")
                    
                    try:
                        os.makedirs(pirate_station_path, exist_ok=True)
                        print(f"✅ Created pirate_station folder at: {pirate_station_path}")
                        return pirate_station_path
                    except PermissionError:
                        print("❌ Permission denied. Try running as administrator/sudo")
                        continue
                    except Exception as e:
                        print(f"❌ Error creating folder: {e}")
                        continue
                else:
                    print("❌ Invalid selection")
                    continue
            except ValueError:
                print("❌ Please enter a valid number")
                continue
        
        elif choice == "3":
            current_dir = os.getcwd()
            pirate_station_path = os.path.join(current_dir, "pirate_station")
            os.makedirs(pirate_station_path, exist_ok=True)
            print(f"✅ Using local pirate_station folder: {pirate_station_path}")
            return pirate_station_path
        
        elif choice == "4":
            print("👋 Goodbye!")
            sys.exit(0)
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4")

def get_drive_free_space(drive_path):
    """Get free space on drive in human readable format."""
    try:
        if platform.system() == "Windows":
            import shutil
            total, used, free = shutil.disk_usage(drive_path)
        else:
            stat = os.statvfs(drive_path)
            free = stat.f_bavail * stat.f_frsize
        
        # Convert to human readable
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if free < 1024.0:
                return f"{free:.1f} {unit} free"
            free /= 1024.0
        return f"{free:.1f} PB free"
    except:
        return "unknown space"

def sanitize_filename(filename):
    """Remove invalid characters from filename."""
    # Remove invalid characters for filesystem
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Remove extra whitespace
    filename = re.sub(r'\s+', ' ', filename).strip()
    # Limit filename length
    if len(filename) > 100:
        filename = filename[:100]
    return filename

def is_playlist_url(url):
    """Check if URL is a playlist."""
    return 'playlist' in url or 'list=' in url

def get_playlist_info(url):
    """Get playlist information without downloading."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,  # Don't extract individual video info
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            if info.get('_type') == 'playlist':
                return {
                    'title': info.get('title', 'Unknown Playlist'),
                    'uploader': info.get('uploader', 'Unknown'),
                    'video_count': len(info.get('entries', [])),
                    'entries': info.get('entries', [])
                }
            else:
                return None
        except Exception as e:
            print(f"Error extracting playlist info: {e}")
            return None

def get_video_info(url):
    """Get video information without downloading."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'uploader': info.get('uploader', 'Unknown'),
                'video_id': info.get('id', '')
            }
        except Exception as e:
            print(f"Error extracting video info: {e}")
            return None

def download_video(url, output_dir):
    """Download video in 1080p MP4 format."""
    print("📹 Downloading video (1080p MP4)...")
    
    ydl_opts = {
        'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'merge_output_format': 'mp4',
        'writeinfojson': False,
        'writesubtitles': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            print("✅ Video download completed!")
            return True
        except Exception as e:
            print(f"❌ Error downloading video: {e}")
            return False

def download_audio(url, output_dir):
    """Download audio in MP3 format."""
    print("🎵 Downloading audio (MP3)...")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            print("✅ Audio download completed!")
            return True
        except Exception as e:
            print(f"❌ Error downloading audio: {e}")
            return False

def download_playlist_videos(url, output_dir):
    """Download all videos in playlist."""
    print("📹 Downloading playlist videos (1080p MP4)...")
    
    ydl_opts = {
        'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]',
        'outtmpl': os.path.join(output_dir, '%(playlist_index)02d - %(title)s.%(ext)s'),
        'merge_output_format': 'mp4',
        'writeinfojson': False,
        'writesubtitles': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            print("✅ Playlist videos download completed!")
            return True
        except Exception as e:
            print(f"❌ Error downloading playlist videos: {e}")
            return False

def download_playlist_audio(url, output_dir):
    """Download all audio from playlist."""
    print("🎵 Downloading playlist audio (MP3)...")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(playlist_index)02d - %(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            print("✅ Playlist audio download completed!")
            return True
        except Exception as e:
            print(f"❌ Error downloading playlist audio: {e}")
            return False

def format_timestamp(seconds):
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def get_transcript(video_id, output_dir, video_title=None):
    """Get transcript from YouTube API."""
    print(f"📝 Fetching transcript{' for ' + video_title if video_title else ''}...")
    
    try:
        # Get available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get English transcript first
        transcript = None
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            # If English not available, get the first available transcript
            try:
                transcript = transcript_list.find_transcript(['en-US', 'en-GB'])
            except:
                # Get any available transcript
                available_transcripts = list(transcript_list)
                if available_transcripts:
                    transcript = available_transcripts[0]
        
        if not transcript:
            print("❌ No transcript available for this video")
            return False
        
        # Fetch the transcript
        transcript_data = transcript.fetch()
        
        # Create two versions of the transcript
        
        # 1. Simple text-only version
        simple_transcript = []
        for entry in transcript_data:
            text = entry.text.strip()  # Use attribute access instead of dictionary
            # Skip music tags and empty entries
            if text and text != '[Music]' and not text.startswith('[') and not text.endswith(']'):
                simple_transcript.append(text)
        
        # 2. Timestamped version
        timestamped_transcript = []
        for entry in transcript_data:
            text = entry.text.strip()  # Use attribute access
            if text:
                timestamp = format_timestamp(entry.start)  # Use attribute access
                timestamped_transcript.append(f"[{timestamp}] {text}")
        
        # Create filenames
        base_name = "transcript"
        if video_title:
            base_name = sanitize_filename(video_title) + "_transcript"
        
        # Save simple transcript
        simple_file = os.path.join(output_dir, f"{base_name}.txt")
        with open(simple_file, 'w', encoding='utf-8') as f:
            f.write(' '.join(simple_transcript))
        
        # Save timestamped transcript
        timestamped_file = os.path.join(output_dir, f"{base_name}_with_timestamps.txt")
        with open(timestamped_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(timestamped_transcript))
        
        print("✅ Transcript downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error fetching transcript: {e}")
        return False

def download_playlist_transcripts(url, output_dir):
    """Download transcripts for all videos in playlist."""
    print("📝 Downloading transcripts for playlist videos...")
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,  # We need full info for video IDs
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            if info.get('_type') == 'playlist':
                entries = info.get('entries', [])
                successful_transcripts = 0
                
                for i, entry in enumerate(entries, 1):
                    if entry:
                        video_id = entry.get('id', '')
                        video_title = entry.get('title', f'Video_{i:02d}')
                        
                        print(f"\n📝 Processing transcript {i}/{len(entries)}: {video_title}")
                        
                        if get_transcript(video_id, output_dir, f"{i:02d} - {video_title}"):
                            successful_transcripts += 1
                
                print(f"\n✅ Downloaded {successful_transcripts}/{len(entries)} transcripts successfully!")
                return True
            else:
                print("❌ Not a playlist URL")
                return False
                
        except Exception as e:
            print(f"❌ Error downloading playlist transcripts: {e}")
            return False

def main():
    print("🎬 YouTube Downloader CLI")
    print("=" * 30)
    
    # Find pirate_station folder first
    base_dir = find_pirate_station_folder()
    
    # Get URL from user
    url = input("\nEnter YouTube URL (video or playlist): ").strip()
    
    if not url:
        print("❌ No URL provided!")
        sys.exit(1)
    
    # Validate URL
    if not ('youtube.com' in url or 'youtu.be' in url):
        print("❌ Please provide a valid YouTube URL!")
        sys.exit(1)
    
    # Check if it's a playlist
    if is_playlist_url(url):
        print("\n🎵 Playlist detected!")
        
        # Get playlist info
        print("🔍 Getting playlist information...")
        playlist_info = get_playlist_info(url)
        
        if not playlist_info:
            print("❌ Could not retrieve playlist information!")
            sys.exit(1)
        
        # Create output directory inside pirate_station
        title = sanitize_filename(playlist_info['title'])
        output_dir = Path(base_dir) / title
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n📁 Output directory: {output_dir}")
        print(f"🎵 Playlist: {playlist_info['title']}")
        print(f"👤 Uploader: {playlist_info['uploader']}")
        print(f"📹 Videos: {playlist_info['video_count']} videos")
        
        # Ask user what to download
        print("\nWhat would you like to download?")
        print("   1. Videos only (MP4)")
        print("   2. Audio only (MP3)")
        print("   3. Both videos and audio")
        print("   4. Everything (videos, audio, and transcripts)")
        
        while True:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                print("\n" + "=" * 50)
                download_playlist_videos(url, output_dir)
                break
            elif choice == "2":
                print("\n" + "=" * 50)
                download_playlist_audio(url, output_dir)
                break
            elif choice == "3":
                print("\n" + "=" * 50)
                download_playlist_videos(url, output_dir)
                print("\n" + "=" * 50)
                download_playlist_audio(url, output_dir)
                break
            elif choice == "4":
                print("\n" + "=" * 50)
                download_playlist_videos(url, output_dir)
                print("\n" + "=" * 50)
                download_playlist_audio(url, output_dir)
                print("\n" + "=" * 50)
                download_playlist_transcripts(url, output_dir)
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4")
    
    else:
        # Single video processing (original functionality)
        print("\n📹 Single video detected!")
        
        # Get video info
        print("🔍 Getting video information...")
        video_info = get_video_info(url)
        
        if not video_info:
            print("❌ Could not retrieve video information!")
            sys.exit(1)
        
        # Create output directory inside pirate_station
        title = sanitize_filename(video_info['title'])
        output_dir = Path(base_dir) / title
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n📁 Output directory: {output_dir}")
        print(f"📹 Video: {video_info['title']}")
        print(f"👤 Uploader: {video_info['uploader']}")
        print(f"⏱️  Duration: {video_info['duration']} seconds")
        
        # Download video
        print("\n" + "=" * 50)
        video_success = download_video(url, output_dir)
        
        # Download audio
        print("\n" + "=" * 50)
        audio_success = download_audio(url, output_dir)
        
        # Get transcript
        print("\n" + "=" * 50)
        get_transcript(video_info['video_id'], output_dir)
    
    print("\n" + "=" * 50)
    print("🎉 Download process completed!")
    print(f"📁 Files saved in: {output_dir.absolute()}")
    
    # List downloaded files
    print("\n📄 Downloaded files:")
    for file in output_dir.iterdir():
        if file.is_file():
            file_size = file.stat().st_size
            if file_size > 1024*1024:  # > 1MB
                size_str = f"{file_size / (1024*1024):.1f} MB"
            elif file_size > 1024:  # > 1KB
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size} B"
            print(f"  • {file.name} ({size_str})")

if __name__ == "__main__":
    # Check if required tools are available
    try:
        import yt_dlp
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("\nPlease install required packages:")
        print("pip install yt-dlp youtube-transcript-api")
        sys.exit(1)
    
    # Check for ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg is required but not found!")
        print("Please install FFmpeg: https://ffmpeg.org/download.html")
        sys.exit(1)
    
    main()
