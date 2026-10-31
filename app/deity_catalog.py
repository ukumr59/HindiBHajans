"""Daily Hindu Deity and Mythological Classic Hit Bhajan Catalog.

Covers:
- Lord Shiva (Mahadev, Rudra, Bholenath)
- Lord Krishna (Govind, Madhav, Radhe Shyam)
- Lord Rama (Raghunandan, Maryada Purushottam)
- Lord Hanuman (Bajrangbali, Sankatmochan, Anjaniputra)
- Goddess Durga (Bhavani, Mahishasura Mardini, Sherawali)
- Goddess Kali (Maha Kali, Kaalratri, Adishakti)

Music style: High-energy modern DJ devotional remixes (128-132 BPM), punchy electronic dhol,
deep 808 sub bass, melodic temple bells, shankha, synthesizer leads, and pristine Hindi/Sanskrit
mantra pronunciation using phonetic syllable transliteration.
"""
from __future__ import annotations

import datetime
from typing import TypedDict


class BhajanThematicEntry(TypedDict):
    deity: str
    scene_file: str
    title_hi: str
    title_en: str
    short_caption: str
    prompt: str
    lyrics: str
    bpm: int


CATALOG: list[BhajanThematicEntry] = [
    {
        "deity": "Lord Shiva",
        "scene_file": "shiva_scene.png",
        "title_hi": "नमो नमो जी शंकरा | हर हर शम्भू डीजे भजन",
        "title_en": "Namo Namo Ji Shankara | Har Har Shambhu Modern DJ Devotional Remix",
        "short_caption": "हर हर महादेव! नमो नमो जी शंकरा और हर हर शम्भू का ऊर्जावान डीजे रीमिक्स 🙏 #Mahadev #Shiva #BhajanAabha",
        "bpm": 128,
        "prompt": (
            "Modern high-energy Indian DJ devotional dance remix, 128 BPM, 4/4, "
            "deep punchy electronic dhol, rolling 808 sub-bass drops, intense temple bells, resonant shankha horn, "
            "catchy festival anthem lead synths, clear melodic studio devotional singing, "
            "flawless authentic Hindi pronunciation of Om Namah Shivaya and Shiva mantras. "
            "NOT low energy, NOT slow ambient."
        ),
        "lyrics": """[Intro]
Om Namah Shivaya... Om Namah Shivaya...
Har Har Mahadev! Jai Bholenath!

[Verse 1]
Namo Namo Ji Shankara, Bholenath Shankara
Hey Triloknath Shambhu, Hey Shivay Shankara
Chandrama Bhaale Viraje, Ganga Jata Mein Sajayi
Damru Baaje Dam Dam Dam, Mahima Tribhuvan Chhayi

[Chorus]
Har Har Shambhu, Har Har Shambhu
Shiv Mahadeva Shambhu
Har Har Shambhu, Har Har Shambhu
Bhole Bhandari Shambhu

[Verse 2]
Neelkanth Vishdhaari Swami, Bhakti Teri Paayi
Bhasma Aarti Karke Bhole, Sabki Vipda Mitayi
Kailashi Avinashi Bhole, Tu Hi Ant Aur Aadi
Sharan Mein Jo Bhi Aaya Prabhu Ke, Bigdi Baat Banadi

[Chorus]
Har Har Shambhu, Har Har Shambhu
Shiv Mahadeva Shambhu
Har Har Shambhu, Har Har Shambhu
Bhole Bhandari Shambhu

[Drop]
Har Har Mahadev... Om Namah Shivaya!

[Outro]
Om Namah Shivaya... Har Har Shambhu...""",
    },
    {
        "deity": "Lord Krishna",
        "scene_file": "krishna_scene.png",
        "title_hi": "अच्युतम केशवम | राधे राधे बरसाने वाली डीजे भजन",
        "title_en": "Achyutam Keshavam | Radhe Radhe Barsane Wali DJ Devotional Remix",
        "short_caption": "जय श्री कृष्णा! अच्युतम केशवम और राधे राधे का मनमोहक डीजे भक्ति रीमिक्स 🙏 #Krishna #RadhaKrishna #BhajanAabha",
        "bpm": 130,
        "prompt": (
            "Modern vibrant Indian DJ devotional dance remix, 130 BPM, 4/4, "
            "infectious electronic dholak groove, melodic bansuri flute leads, thumping bass drops, "
            "uplifting stadium chorus, clear melodic vocal enunciation, "
            "crystal clear Sanskrit-Hindi pronunciation of Achyutam Keshavam and Hare Krishna. "
            "NOT acoustic only, NOT mournful."
        ),
        "lyrics": """[Intro]
Hare Krishna Hare Rama... Radhe Radhe!
Govind Bolo Hari Gopal Bolo...

[Verse 1]
Achyutam Keshavam Krishna Damodaram
Rama Narayanam Janaki Vallabham
Kaun Kehte Hain Bhagwan Aate Nahin
Gopiyon Ki Tarah Hum Bulate Nahin

[Chorus]
Radhe Radhe, Radhe Radhe
Radhe Barsane Wali Radhe
Shyam Se Milade Radhe
Mere Krishna Murari Radhe

[Verse 2]
Makhan Chor Murari Mohan, Bansuri Bajaye
Yamuna Ke Teere Prabhu Ji, Sabka Mann Harshaye
Choti Choti Gaiya Chote Chote Gwal
Choto So Mero Madan Gopal

[Chorus]
Radhe Radhe, Radhe Radhe
Radhe Barsane Wali Radhe
Shyam Se Milade Radhe
Mere Krishna Murari Radhe

[Drop]
Hare Krishna Hare Krishna, Krishna Krishna Hare Hare...
Hare Rama Hare Rama, Rama Rama Hare Hare...

[Outro]
Radhe Radhe... Jai Shri Krishna!""",
    },
    {
        "deity": "Lord Hanuman",
        "scene_file": "hanuman_scene.png",
        "title_hi": "दुनिया चले ना श्री राम के बिना | जय बजरंगबली डीजे भजन",
        "title_en": "Duniya Chale Na Shri Ram Ke Bina | Jai Bajrangbali Power DJ Bhajan",
        "short_caption": "जय वीर बजरंगी! संकटमोचन श्री हनुमान का शक्तिशाली डीजे भक्ति गान 🙏 #Hanuman #Bajrangbali #BhajanAabha",
        "bpm": 132,
        "prompt": (
            "High-octane stadium-energy Indian DJ devotional remix, 132 BPM, 4/4, "
            "thunderous nagada and punjabi dhol, brass fanfare drops, heavy driving synth sub-bass, "
            "heroic triumphant devotional vocals, "
            "flawless Hindi pronunciation of Jai Bajrangbali and Hanuman mantras. "
            "NOT dull, NOT quiet."
        ),
        "lyrics": """[Intro]
Jai Shri Ram! Jai Veer Bajrangi!
Pawansut Hanuman Ki Jai!

[Verse 1]
Duniya Chale Na Shri Ram Ke Bina
Ram Ji Chale Na Hanuman Ke Bina
Sindoor Chola Tan Par Sohe, Gada Haath Mein Dhari
Sankat Harne Wale Swami, Kripa Karo Kripakari

[Chorus]
Jai Bajrangbali, Jai Jai Bajrangbali
Sankat Mochan Naam Tiharo, Jai Bajrangbali
Jai Bajrangbali, Jai Jai Bajrangbali
Ram Bhakt Shiromani, Jai Bajrangbali

[Verse 2]
Lanka Jalai Pal Mein Prabhu Ne, Sur Asur Balihari
Sanjivani Booti Laye Swami, Lakshman Praan Ubari
Jahan Jahan Kirtan Ram Ka Hove, Wahan Wahan Tum Padhare
Haath Jodkar Kare Vandana, Bhakton Ke Sankat Taare

[Chorus]
Jai Bajrangbali, Jai Jai Bajrangbali
Sankat Mochan Naam Tiharo, Jai Bajrangbali
Jai Bajrangbali, Jai Jai Bajrangbali
Ram Bhakt Shiromani, Jai Bajrangbali

[Drop]
Jai Jai Shri Ram... Bolo Bajrangbali Ki Jai!

[Outro]
Om Hanumate Namah... Jai Bajrangbali!""",
    },
    {
        "deity": "Lord Rama",
        "scene_file": "rama_scene.png",
        "title_hi": "रघुपति राघव राजा राम | राम आएंगे डीजे भजन",
        "title_en": "Raghupati Raghav Raja Ram | Ram Aayenge Majestic DJ Bhajan",
        "short_caption": "जय श्री राम! रघुपति राघव राजा राम और राम आएंगे का दिव्य डीजे रीमिक्स 🙏 #ShriRam #Ayodhya #BhajanAabha",
        "bpm": 128,
        "prompt": (
            "Majestic modern Indian DJ devotional anthem, 128 BPM, 4/4, "
            "ceremonial dhol beats, soaring brass horns, electronic pumping bassline, "
            "uplifting stadium devotional chorus, "
            "pristine clear pronunciation of Raghupati Raghav and Shri Ram. "
            "NOT slow, NOT ambient."
        ),
        "lyrics": """[Intro]
Shri Ram Jai Ram Jai Jai Ram...
Siyavar Ramchandra Ki Jai!

[Verse 1]
Raghupati Raghav Raja Ram
Patit Pavan Sita Ram
Sundar Chhabi Man Bhavani
Ayodhya Ke Raj Dulare

[Chorus]
Ram Aayenge, Ram Aayenge
Mere Ram Lalla Aayenge
Raghupati Raghav Raja Ram
Patit Pavan Sita Ram

[Verse 2]
Mangal Bhavan Amangal Hari
Dravahu Su Dasharath Ajir Bihari
Maryada Purushottam Swami, Karuna Ke Sagar
Charan Kamal Mein Sheesh Nawate, Jai Kripa Ke Aagar

[Chorus]
Ram Aayenge, Ram Aayenge
Mere Ram Lalla Aayenge
Raghupati Raghav Raja Ram
Patit Pavan Sita Ram

[Drop]
Jai Shri Ram! Ayodhya Pati Ramchandra Ki Jai!

[Outro]
Shri Ram Jai Ram Jai Jai Ram... Jai Shri Ram!""",
    },
    {
        "deity": "Goddess Durga",
        "scene_file": "durga_scene.png",
        "title_hi": "चलो बुलावा आया है | जय माँ दुर्गे डीजे भजन",
        "title_en": "Chalo Bulawa Aaya Hai | Jai Maa Durge Navratri DJ Dance Bhajan",
        "short_caption": "जय माता दी! माँ दुर्गा का सबसे प्रसिद्ध भक्ति डीजे गान 🙏 #MaaDurga #Navratri #BhajanAabha",
        "bpm": 130,
        "prompt": (
            "Electrifying festive DJ Navratri devotional remix, 130 BPM, 4/4, "
            "energetic dhak and dhol drums, resonant trishul chimes, pumping electronic dance drops, "
            "celebratory festive vocals, crisp Sanskrit-Hindi diction of Jai Mata Di and Maa Durge. "
            "NOT soft, NOT timid."
        ),
        "lyrics": """[Intro]
Jai Mata Di! Jai Maa Jagdambe!
Sarva Mangala Mangalye Shive Sarvartha Sadhike...

[Verse 1]
Chalo Bulawa Aaya Hai, Mata Ne Bulaya Hai
Unche Parvat Par Baithe Hai, Sherawali Maiya
Ashta Bhuja Mein Astra Saje Hain, Karti Sabki Bhalayi
Laal Chunariya Sheesh Pe Sohe, Jagat Kalyani Aayi

[Chorus]
Jai Maa Ambe, Jai Maa Durge
Jai Jai Bhavani, Jai Maa Durge
Jai Maa Ambe, Jai Maa Durge
Bhakton Ki Rakshak, Jai Maa Durge

[Verse 2]
Mahishasur Mardini Maiya, Asur Nikandan Haari
Teri Kripa Se Roshan Hoti, Yeh Duniya Sari
Jholi Bhar De Sabki Mata, Dware Tere Aaye
Jo Bhi Teri Sharan Mein Aaye, Bhav Se Paar Ho Jaye

[Chorus]
Jai Maa Ambe, Jai Maa Durge
Jai Jai Bhavani, Jai Maa Durge
Jai Maa Ambe, Jai Maa Durge
Bhakton Ki Rakshak, Jai Maa Durge

[Drop]
Bolo Saanche Darbar Ki Jai... Jai Mata Di!

[Outro]
Jai Mata Di... Jai Maa Durge...""",
    },
    {
        "deity": "Goddess Kali",
        "scene_file": "kali_scene.png",
        "title_hi": "महाकाली खप्पर वाली | ॐ जयंती मंगला काली डीजे भजन",
        "title_en": "Mahakali Khappar Wali | Om Jayanti Mangala Kali EDM Bass Bhajan",
        "short_caption": "जय माँ महाकाली! माँ कालरात्रि का प्रचंड स्तुति डीजे भक्ति गान 🙏 #MahaKali #Kaalratri #BhajanAabha",
        "bpm": 132,
        "prompt": (
            "Thunderous intense Indian EDM bass devotional track, 132 BPM, 4/4, "
            "rapid war dholak, deep sub-bass vibrations, sharp temple bells, "
            "hypnotic energetic vocals, immaculate mantra enunciation of Om Kreem Kalikayai Namah. "
            "NOT slow, NOT dull."
        ),
        "lyrics": """[Intro]
Om Kreem Kalikayai Namah...
Om Jayanti Mangala Kali Bhadrakali Kapalini...

[Verse 1]
Khappar Wali Khadga Viraje, Kaalratri Mata
Dusht Damani Shaktishali, Tu Hi Bhagya Vidhata
Mundmal Gale Mein Sohe, Cham Cham Chamkanta
Bhakton Ke Dukh Harne Wali, Mahima Ananta

[Chorus]
Jai Mahakali, Jai Bhadrakali
Sankat Harini, Khappar Wali
Jai Mahakali, Jai Bhadrakali
Adishakti Mata Hamari

[Verse 2]
Raktbeej Ka Maan Mitaya, Chund Mund Sanhare
Maa Ke Krodh Se Doli Dharti, Sur Aur Asur Hare
Shiv Ji Ke Urr Par Paon Dhari, Shant Hui Mahamaya
Teri Sharan Mein Aakar Humne, Amrit Jeevan Paya

[Chorus]
Jai Mahakali, Jai Bhadrakali
Sankat Harini, Khappar Wali
Jai Mahakali, Jai Bhadrakali
Adishakti Mata Hamari

[Drop]
Om Jayanti Mangala Kali Bhadrakali Kapalini...
Durga Kshama Shiva Dhatri Swaha Swadha Namostute!

[Outro]
Jai Maa Kali... Jai Mahakali!""",
    },
]


def get_daily_bhajan(day_offset: int = 0) -> BhajanThematicEntry:
    """Return the deterministic deity bhajan entry based on the current calendar day."""
    now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=5.5)  # IST
    day_index = (now.toordinal() + day_offset) % len(CATALOG)
    return CATALOG[day_index]
