"""Daily Hindu Deity and Mythological Classic Hit Bhajan Catalog.

Covers:
- Lord Shiva (Mahadev, Rudra, Bholenath)
- Lord Krishna (Govind, Madhav, Radhe Shyam)
- Lord Rama (Raghunandan, Maryada Purushottam)
- Lord Hanuman (Bajrangbali, Sankatmochan, Anjaniputra)
- Goddess Durga (Bhavani, Mahishasura Mardini, Sherawali)
- Goddess Kali (Maha Kali, Kaalratri, Adishakti)

Music style: High-energy modern DJ devotional remixes (128-132 BPM), punchy electronic dhol,
deep 808 sub bass, melodic temple bells, shankha, synthesizer leads, and 100% authentic Hindi/Sanskrit
mantra pronunciation using pure Devanagari script for ACE-Step v1.5 phonetic precision.
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
        "scene_file": "singer_image.png",
        "title_hi": "नमो नमो जी शंकरा | हर हर शम्भू डीजे भजन",
        "title_en": "Namo Namo Ji Shankara | Har Har Shambhu Modern DJ Devotional Remix",
        "short_caption": "हर हर महादेव! नमो नमो जी शंकरा और हर हर शम्भू का ऊर्जावान डीजे रीमिक्स 🙏 #Mahadev #Shiva #BhajanAabha",
        "bpm": 128,
        "prompt": (
            "Modern high-energy Indian DJ devotional dance remix, 128 BPM, 4/4, "
            "deep punchy electronic dhol, rolling 808 sub-bass drops, intense temple bells, resonant shankha horn, "
            "catchy festival anthem lead synths, authentic clear Indian male devotional singing voice, "
            "100% accurate Hindi and Sanskrit pronunciation of Om Namah Shivaya and Shiva mantras, "
            "studio quality, flawless diction. NOT low energy, NOT slow ambient."
        ),
        "lyrics": """[intro]
ॐ नमः शिवाय... ॐ नमः शिवाय...
हर हर महादेव! जय भोलेनाथ!

[verse 1]
नमो नमो जी शंकरा, भोलेनाथ शंकरा
हे त्रिलोकनाथ शम्भू, हे शिवाय शंकरा
चन्द्रमा भाले विराजे, गंगा जटा में सजाई
डमरू बाजे डम डम डम, महिमा त्रिभुवन छाई

[chorus]
हर हर शम्भू, हर हर शम्भू
शिव महादेवा शम्भू
हर हर शम्भू, हर हर शम्भू
भोले भण्डारी शम्भू

[verse 2]
नीलकंठ विषधारी स्वामी, भक्ति तेरी पाई
भस्म आरती करके भोले, सबकी विपदा मिटाई
कैलाशी अविनाशी भोले, तू ही अंत और आदी
शरण में जो भी आया प्रभु के, बिगड़ी बात बना दी

[chorus]
हर हर शम्भू, हर हर शम्भू
शिव महादेवा शम्भू
हर हर शम्भू, हर हर शम्भू
भोले भण्डारी शम्भू

[drop]
हर हर महादेव... ॐ नमः शिवाय!

[outro]
ॐ नमः शिवाय... हर हर शम्भू...""",
    },
    {
        "deity": "Lord Krishna",
        "scene_file": "singer_image.png",
        "title_hi": "अच्युतम केशवम | राधे राधे बरसाने वाली डीजे भजन",
        "title_en": "Achyutam Keshavam | Radhe Radhe Barsane Wali DJ Devotional Remix",
        "short_caption": "जय श्री कृष्णा! अच्युतम केशवम और राधे राधे का मनमोहक डीजे भक्ति रीमिक्स 🙏 #Krishna #RadhaKrishna #BhajanAabha",
        "bpm": 130,
        "prompt": (
            "Modern vibrant Indian DJ devotional dance remix, 130 BPM, 4/4, "
            "infectious electronic dholak groove, melodic bansuri flute leads, thumping bass drops, "
            "uplifting stadium chorus, authentic clear Indian male devotional singing voice, "
            "100% accurate Sanskrit and Hindi pronunciation of Achyutam Keshavam and Hare Krishna Mahamantra, "
            "perfect diction, temple studio clarity. NOT acoustic only, NOT mournful."
        ),
        "lyrics": """[intro]
हरे कृष्णा हरे रामा... राधे राधे!
गोविन्द बोलो हरि गोपाल बोलो...

[verse 1]
अच्युतं केशवं कृष्ण दामोदरं
राम नारायणं जानकी वल्लभम्
कौन कहते हैं भगवान आते नहीं
गोपियों की तरह हम बुलाते नहीं

[chorus]
राधे राधे, राधे राधे
राधे बरसाने वाली राधे
श्याम से मिला दे राधे
मेरे कृष्ण मुरारी राधे

[verse 2]
माखन चोर मुरारी मोहन, बाँसुरी बजाए
यमुना के तीरे प्रभु जी, सबका मन हरषाए
छोटी छोटी गैया छोटे छोटे ग्वाल
छोटो सो मेरो मदन गोपाल

[chorus]
राधे राधे, राधे राधे
राधे बरसाने वाली राधे
श्याम से मिला दे राधे
मेरे कृष्ण मुरारी राधे

[drop]
हरे कृष्ण हरे कृष्ण, कृष्ण कृष्ण हरे हरे...
हरे राम हरे राम, राम राम हरे हरे...

[outro]
राधे राधे... जय श्री कृष्णा!""",
    },
    {
        "deity": "Lord Hanuman",
        "scene_file": "singer_image.png",
        "title_hi": "दुनिया चले ना श्री राम के बिना | जय बजरंगबली डीजे भजन",
        "title_en": "Duniya Chale Na Shri Ram Ke Bina | Jai Bajrangbali Power DJ Bhajan",
        "short_caption": "जय वीर बजरंगी! संकटमोचन श्री हनुमान का शक्तिशाली डीजे भक्ति गान 🙏 #Hanuman #Bajrangbali #BhajanAabha",
        "bpm": 132,
        "prompt": (
            "High-octane stadium-energy Indian DJ devotional remix, 132 BPM, 4/4, "
            "thunderous nagada and punjabi dhol, brass fanfare drops, heavy driving synth sub-bass, "
            "heroic triumphant devotional vocals, authentic Indian male singing voice, "
            "100% accurate Hindi and Sanskrit pronunciation of Jai Bajrangbali and Hanuman stuti, "
            "flawless clear diction, studio quality. NOT dull, NOT quiet."
        ),
        "lyrics": """[intro]
जय श्री राम! जय वीर बजरंगी!
पवनसुत हनुमान की जय!

[verse 1]
दुनिया चले ना श्री राम के बिना
राम जी चलें ना हनुमान के बिना
सिन्दूर चोला तन पर सोहे, गदा हाथ में धारी
संकट हरने वाले स्वामी, कृपा करो कृपाकारी

[chorus]
जय बजरंगबली, जय जय बजरंगबली
संकट मोचन नाम तिहारो, जय बजरंगबली
जय बजरंगबली, जय जय बजरंगबली
राम भक्त शिरोमणि, जय बजरंगबली

[verse 2]
लंका जलाई पल में प्रभु ने, सुर असुर बलिहारी
संजीवनी बूटी लाये स्वामी, लक्ष्मण प्राण उबारी
जहाँ जहाँ कीर्तन राम का होवे, तहाँ तहाँ तुम पधारे
हाथ जोड़कर करें वंदना, भक्तों के संकट टारे

[chorus]
जय बजरंगबली, जय जय बजरंगबली
संकट मोचन नाम तिहारो, जय बजरंगबली
जय बजरंगबली, जय जय बजरंगबली
राम भक्त शिरोमणि, जय बजरंगबली

[drop]
जय जय श्री राम... बोलो बजरंगबली की जय!

[outro]
ॐ हनुमते नमः... जय बजरंगबली!""",
    },
    {
        "deity": "Lord Rama",
        "scene_file": "singer_image.png",
        "title_hi": "रघुपति राघव राजा राम | राम आएंगे डीजे भजन",
        "title_en": "Raghupati Raghav Raja Ram | Ram Aayenge Majestic DJ Bhajan",
        "short_caption": "जय श्री राम! रघुपति राघव राजा राम और राम आएंगे का दिव्य डीजे रीमिक्स 🙏 #ShriRam #Ayodhya #BhajanAabha",
        "bpm": 128,
        "prompt": (
            "Majestic modern Indian DJ devotional anthem, 128 BPM, 4/4, "
            "ceremonial dhol beats, soaring brass horns, electronic pumping bassline, "
            "uplifting stadium devotional chorus, authentic Indian male devotional singing voice, "
            "100% accurate Hindi and Sanskrit pronunciation of Raghupati Raghav, Ram Aayenge and Ramcharitmanas chaupai, "
            "crystal clear studio diction. NOT slow, NOT ambient."
        ),
        "lyrics": """[intro]
श्री राम जय राम जय जय राम...
सियावर रामचंद्र की जय!

[verse 1]
रघुपति राघव राजा राम
पतित पावन सीता राम
सुन्दर छवि मन भावनी
अयोध्या के राज दुलारे

[chorus]
राम आएंगे, राम आएंगे
मेरे राम लला आएंगे
रघुपति राघव राजा राम
पतित पावन सीता राम

[verse 2]
मंगल भवन अमंगल हारी
द्रवहु सुदशरथ अजिर बिहारी
मर्यादा पुरुषोत्तम स्वामी, करुणा के सागर
चरण कमल में शीश नवाते, जय कृपा के आगार

[chorus]
राम आएंगे, राम आएंगे
मेरे राम लला आएंगे
रघुपति राघव राजा राम
पतित पावन सीता राम

[drop]
जय श्री राम! अयोध्यापति रामचंद्र की जय!

[outro]
श्री राम जय राम जय जय राम... जय श्री राम!""",
    },
    {
        "deity": "Goddess Durga",
        "scene_file": "singer_image.png",
        "title_hi": "चलो बुलावा आया है | जय माँ दुर्गे डीजे भजन",
        "title_en": "Chalo Bulawa Aaya Hai | Jai Maa Durge Navratri DJ Dance Bhajan",
        "short_caption": "जय माता दी! माँ दुर्गा का सबसे प्रसिद्ध भक्ति डीजे गान 🙏 #MaaDurga #Navratri #BhajanAabha",
        "bpm": 130,
        "prompt": (
            "Electrifying festive DJ Navratri devotional remix, 130 BPM, 4/4, "
            "energetic dhak and dhol drums, resonant trishul chimes, pumping electronic dance drops, "
            "celebratory festive vocals, authentic Indian male devotional singing, "
            "100% accurate Hindi and Sanskrit pronunciation of Jai Mata Di and Sarva Mangala Mangalye shloka, "
            "studio quality, crystal clear diction. NOT soft, NOT timid."
        ),
        "lyrics": """[intro]
जय माता दी! जय माँ जगदम्बे!
सर्वमङ्गलमाङ्गल्ये शिवे सर्वार्थसाधिके...

[verse 1]
चलो बुलावा आया है, माता ने बुलाया है
ऊँचे पर्वत पर बैठी हैं, शेरावाली मैया
अष्ट भुजा में अस्त्र सजे हैं, करती सबकी भलाई
लाल चुनरिया शीश पे सोहे, जगत कल्याणी आई

[chorus]
जय माँ अम्बे, जय माँ दुर्गे
जय जय भवानी, जय माँ दुर्गे
जय माँ अम्बे, जय माँ दुर्गे
भक्तों की रक्षक, जय माँ दुर्गे

[verse 2]
महिषासुर मर्दिनी मैया, असुर निकन्दन हारी
तेरी कृपा से रोशन होती, यह दुनिया सारी
झोली भर दे सबकी माता, द्वारे तेरे आए
जो भी तेरी शरण में आए, भव से पार हो जाए

[chorus]
जय माँ अम्बे, जय माँ दुर्गे
जय जय भवानी, जय माँ दुर्गे
जय माँ अम्बे, जय माँ दुर्गे
भक्तों की रक्षक, जय माँ दुर्गे

[drop]
बोलो सांचे दरबार की जय... जय माता दी!

[outro]
जय माता दी... जय माँ दुर्गे...""",
    },
    {
        "deity": "Goddess Kali",
        "scene_file": "singer_image.png",
        "title_hi": "महाकाली खप्पर वाली | ॐ जयंती मंगला काली डीजे भजन",
        "title_en": "Mahakali Khappar Wali | Om Jayanti Mangala Kali EDM Bass Bhajan",
        "short_caption": "जय माँ महाकाली! माँ कालरात्रि का प्रचंड स्तुति डीजे भक्ति गान 🙏 #MahaKali #Kaalratri #BhajanAabha",
        "bpm": 132,
        "prompt": (
            "Thunderous intense Indian EDM bass devotional track, 132 BPM, 4/4, "
            "rapid war dholak, deep sub-bass vibrations, sharp temple bells, "
            "hypnotic energetic devotional vocals, authentic Indian male singing voice, "
            "100% accurate Sanskrit mantra pronunciation of Om Kreem Kalikayai Namah and Argala Stotram shlokas, "
            "powerful studio diction, crystal clear. NOT slow, NOT dull."
        ),
        "lyrics": """[intro]
ॐ क्रीं कालिकायै नमः...
ॐ जयन्ती मङ्गला काली भद्रकाली कपालिनी...

[verse 1]
खप्पर वाली खड्ग विराजे, कालरात्रि माता
दुष्ट दमनी शक्तिशाली, तू ही भाग्य विधाता
मुंडमाल गले में सोहे, चम चम चमकन्ता
भक्तों के दुःख हरने वाली, महिमा अनन्ता

[chorus]
जय महाकाली, जय भद्रकाली
संकट हारिणी, खप्पर वाली
जय महाकाली, जय भद्रकाली
आदिशक्ति माता हमारी

[verse 2]
रक्तबीज का मान मिटाया, चण्ड मुण्ड संहारे
माँ के क्रोध से डोली धरती, सुर और असुर हारे
शिव जी के उर पर पाँव धरी, शांत हुई महामाया
तेरी शरण में आकर हमने, अमृत जीवन पाया

[chorus]
जय महाकाली, जय भद्रकाली
संकट हारिणी, खप्पर वाली
जय महाकाली, जय भद्रकाली
आदिशक्ति माता हमारी

[drop]
ॐ जयन्ती मङ्गला काली भद्रकाली कपालिनी...
दुर्गा क्षमा शिवा धात्री स्वाहा स्वधा नमोऽस्तु ते!

[outro]
जय माँ काली... जय महाकाली!""",
    },
]


def get_daily_bhajan(day_offset: int = 0) -> BhajanThematicEntry:
    """Return the deterministic deity bhajan entry based on the current calendar day."""
    now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=5.5)  # IST
    day_index = (now.toordinal() + day_offset) % len(CATALOG)
    return CATALOG[day_index]
