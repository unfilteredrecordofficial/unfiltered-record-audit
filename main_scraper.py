import os
import requests
import json
from bs4 import BeautifulSoup
from groq import Groq
from google.cloud import texttospeech
from pydub import AudioSegment
from datetime import datetime

# --- 1. CONFIGURATION & DATA ACQUISITION ---

def fetch_latest_bill_data():
    """Fetches the latest bill data from the Congress.gov API."""
    congress_key = os.environ.get('CONGRESS_KEY')
    if not congress_key:
        raise ValueError("CONGRESS_KEY not found in environment variables.")

    # Sorts by latest action date to get the most recent bill
    api_url = f"https://api.congress.gov/v3/bill?api_key={congress_key}&sort=latestActionDate&limit=1&format=json"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        
        # Safely retrieve the latest bill
        latest_bill = data['bills'][0] if data and 'bills' in data and data['bills'] else None
        return latest_bill
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Congress.gov data: {e}")
        return None

def scrape_cbo_cost_estimate(bill_number_data):
    """Scrapes the CBO website for the cost estimate of a given bill."""
    # Build a predictable CBO URL based on the bill type and number
    bill_type = bill_number_data.get('billType', 'hr').lower()
    bill_number = bill_number_data.get('number', '1')
    
    # NOTE: CBO URL structure is subject to change; requires maintenance.
    cbo_url = f"https://www.cbo.gov/publication/cost-estimate/{bill_type}/{bill_number}"

    try:
        response = requests.get(cbo_url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # This selector is a common target for the brief summary text on CBO pages.
        # It needs verification based on current CBO website structure.
        summary_element = soup.find('div', class_='field-name-title') 
        
        if summary_element:
            return summary_element.get_text(strip=True)
        else:
            # Fallback if specific element is not found
            return "CBO cost estimate summary paragraph not found."

    except requests.exceptions.RequestException as e:
        return f"Error scraping CBO: Could not access {cbo_url}. {e}"

# --- 2. AI NARRATIVE ENGINE ---

def generate_podcast_script(bill_data, cbo_cost_text):
    """Generates the podcast script using the Groq API."""
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    client = Groq(api_key=groq_api_key)

    # The System Prompt is the CRITICAL security and style guardrail
    system_prompt = (
        "You are 'The Unfiltered Record' AI host. Your role is to deliver a strictly non-partisan, 90-second "
        "legislative audit. The script MUST have three distinct sections: 1. A neutral introduction of the bill "
        "and its sponsor. 2. An unbiased summary of the bill's main provisions and the CBO's official cost estimate. "
        "3. A concise, neutral closing statement. The total script length must be approximately 250 words. "
        "Maintain an authoritative, factual, and neutral tone. Do not include greetings or external commentary."
    )
    
    # Construct the user message with all acquired data
    bill_title = bill_data.get('title', 'N/A')
    sponsor_name = bill_data.get('sponsor', {}).get('fullName', 'Unknown Sponsor')
    
    bill_details = (
        f"Generate a 90-second script based on the following facts:\n\n"
        f"Bill Title: {bill_title}\n"
        f"Bill Sponsor: {sponsor_name}\n"
        f"CBO Cost Analysis Summary: {cbo_cost_text}"
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": bill_details}
            ],
            model="mixtral-8x7b-32768", # Optimal balance of speed and quality
            temperature=0.4 # Lower temperature for factual consistency
        )
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating Groq script: {e}")
        return "The AI script failed to generate due to an API error."


# --- 3. AUDIO PRODUCTION & FINALIZATION ---

def generate_tts_audio(script_text, output_file="podcast_script.mp3"):
    """Generates audio file using GCP TTS (authenticates via WIF)."""
    # The TextToSpeechClient automatically uses the short-lived credentials 
    # established by the 'google-github-actions/auth' step.
    tts_client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=script_text)

    # Using a Neural2 voice for high quality (WaveNet/Neural2)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Neural2-J"  # Example: High-quality female voice
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open(output_file, "wb") as out:
        out.write(response.audio_content)
    
    return output_file

def stitch_podcast_segments(script_audio_path):
    """Stitches intro music to the generated script audio using pydub/FFmpeg."""
    try:
        # Load audio segments
        intro = AudioSegment.from_mp3("intro_music.mp3")
        script_audio = AudioSegment.from_mp3(script_audio_path)
        
        # Apply professional touches: duck the music and stitch
        intro_duration_ms = 10000  # Use 10 seconds of intro music
        intro_segment = intro[:intro_duration_ms].fade_out(1500)
        
        final_podcast = intro_segment + script_audio
        
        # Export with date for file management
        today_date = datetime.now().strftime('%Y%m%d')
        final_podcast_path = f"The_Unfiltered_Record_{today_date}.mp3"
        
        # Export the final file
        final_podcast.export(final_podcast_path, format="mp3")
        
        return final_podcast_path
    
    except FileNotFoundError:
        print("ERROR: 'intro_music.mp3' not found or FFmpeg is not installed/configured.")
        return script_audio_path # Return unstitched audio as fallback


# --- MAIN EXECUTION ---

def main():
    """Executes the entire automated podcast pipeline."""
    
    # 1. DATA ACQUISITION 
    print("--- STEP 1: Starting Data Acquisition ---")
    
    bill_data = fetch_latest_bill_data()
    if not bill_data:
        print("FATAL: Could not fetch bill data. Pipeline halted.")
        return
        
    print(f"-> Acquired Bill: {bill_data.get('title', 'N/A')}")

    cbo_cost_text = scrape_cbo_cost_estimate(bill_data)
    print(f"-> CBO Cost Summary: {cbo_cost_text[:75]}...")

    # 2. AI NARRATIVE ENGINE 
    print("\n--- STEP 2: Generating Podcast Script (Groq) ---")
    podcast_script = generate_podcast_script(bill_data, cbo_cost_text)
    print(f"-> Script Generated. Length: {len(podcast_script)} characters.")

    # 3. AUDIO PRODUCTION & FINALIZATION 
    print("\n--- STEP 3: Synthesizing Audio (GCP TTS/WIF) ---")
    script_audio_path = generate_tts_audio(podcast_script)
    print(f"-> TTS audio saved to {script_audio_path}")
    
    final_podcast_path = stitch_podcast_segments(script_audio_path)
    print(f"\n--- PIPELINE COMPLETE! ---")
    print(f"SUCCESS: Final podcast MP3 ready at: {final_podcast_path}")
    
    # --- FUTURE STEP 4: PUBLISHING ---
    # The next step would be integrating your final file upload/publishing logic here.


if __name__ == "__main__":
    main()
