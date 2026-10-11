"""Daily Hindu Deity and Mythological Bhajan Catalog.

Covers:
- Lord Shiva (Mahadev, Rudra, Bholenath)
- Lord Krishna (Govind, Madhav, Radhe Shyam)
- Lord Rama (Raghunandan, Maryada Purushottam)
- Lord Hanuman (Bajrangbali, Sankatmochan, Anjaniputra)
- Goddess Durga (Bhavani, Mahishasura Mardini, Sherawali)
- Goddess Kali (Maha Kali, Kaalratri, Adishakti)

Music style: High-energy modern devotional, 128-132 BPM, powerful DJ-friendly drops,
punchy dholak/dhol, nagada, temple bells, shankha, and modern bassline synthesis.
Zero-cost, fully autonomous generation for Bhajan Aabha.
"""
from __future__ import annotations

import datetime
from typing import TypedDict


class BhajanThematicEntry(TypedDict):
    deity: str
    title_hi: str
    title_en: str
    short_caption: str
    prompt: str
    lyrics: str
    bpm: int


CATALOG: list[BhajanThematicEntry] = [
    {
        "deity": "Lord Shiva",
        "title_hi": "हे महाकाल शम्भू | तांडव स्तोत्रम भाव",
        "title_en": "Hey Mahakal Shambhu | Shiva Tandav Devotional EDM Bhajan",
        "short_caption": "हर हर महादेव! भस्म आरती और तांडव भाव का शक्तिशाली शिव भजन 🙏 #Mahadev #Shiva #BhajanAabha",
        "bpm": 128,
        "prompt": (
            "Modern high-energy Hindi devotional Shiva bhajan, 128 BPM, 4/4, powerful dynamic male devotional vocal, "
            "deep punchy dhol, intense temple bells, resonant shankha horn, electronic bass drops, melodic damru rhythms, "
            "catchy anthem chorus, polished club and DJ production, crisp and clean Hindi pronunciation. "
            "NOT ambient, NOT low energy, NOT slow meditation."
        ),
        "lyrics": """[Intro]
ॐ नमः शिवाय... हर हर महादेव...
जय शम्भू... जय महाकाल...

[Verse 1]
जटा कटा हसंभ्रम भ्रमन्निलिम्पनिर्झरी
माथे पर चंदा विराजे, त्रिशूल धरे त्रिपुरारी
डमरू बाजे घनघनाए, डम डम डम डम डमरू बोले
भक्तों के संकट को हरने, शिव शंकर का डमरू डोले

[Chorus]
हर हर शम्भू, जय जय शम्भू
हे त्रिपुरारी, संकटहारी शम्भू
हर हर शम्भू, जय जय शम्भू
मेरे मन में बसते भोले शम्भू

[Verse 2]
गंगा की पावन धारा बहती, नीलकंठ विषधारी
भस्म रमाए बैठे नंदी, महिमा त्रिभुवन भारी
कैलाशी अविनाशी स्वामी, तू ही अंत अरु आदि
तेरी शरण में आया जो भी, बिगड़ी बात बना दी

[Chorus]
हर हर शम्भू, जय जय शम्भू
हे त्रिपुरारी, संकटहारी शम्भू
हर हर शम्भू, जय जय शम्भू
मेरे मन में बसते भोले शम्भू

[Drop]
ॐ नमः शिवाय... बम बम भोले...

[Outro]
हर हर महादेव... शम्भू... ॐ नमः शिवाय...""",
    },
    {
        "deity": "Lord Hanuman",
        "title_hi": "बजरंग बाण गूंजे | संकटमोचन हनुमान भजन",
        "title_en": "Bajrangbali Jai Jai Hanuman | High Energy DJ Devotional Bhajan",
        "short_caption": "जय वीर बजरंगी! संकटमोचन श्री हनुमान का ऊर्जावान भजन 🙏 #Hanuman #Bajrangbali #BhajanAabha",
        "bpm": 130,
        "prompt": (
            "Modern energetic Hindi devotional Hanuman bhajan, 130 BPM, 4/4, heroic and thunderous male devotional vocals, "
            "thundering nagada and punjabi dhol, temple bells, brass horns, driving rhythmic synthesizer bass, "
            "soaring anthem chorus, stadium energy, perfect Hindi pronunciation. "
            "NOT low energy, NOT acoustic-only."
        ),
        "lyrics": """[Intro]
जय बजरंगबली... जय श्री राम...
पवनसुत हनुमान की जय...

[Verse 1]
अतुलित बलधामा हे स्वामी, अंजनी के नंदन प्यारे
राम काज संवारने वाले, भक्तों के कष्ट निवारे
सिंदूरी चोला तन सोहे, गदा हाथ विकराली
लंका जलाई पल भर में, सुर-असुर सब बलिहारी

[Chorus]
जय बजरंगी, संकट भंजन
हे महावीर, हे राम दुलारे
जय बजरंगी, संकट भंजन
काटो कष्ट हमारे प्यारे

[Verse 2]
लक्ष्मण के प्राण बचाए, बूटी ले आए संजीवन
राम नाम उर में जो ध्याए, पावन हो जाए जीवन
भूत पिसाच निकट नहिं आवै, महावीर जब नाम सुनावै
तेरी कृपा से राम मिलेंगे, हर संकट कट जावै

[Chorus]
जय बजरंगी, संकट भंजन
हे महावीर, हे राम दुलारे
जय बजरंगी, संकट भंजन
काटो कष्ट हमारे प्यारे

[Drop]
जय श्री राम... जय वीर हनुमान...

[Outro]
सियापति रामचंद्र की जय... पवनसुत हनुमान की जय...""",
    },
    {
        "deity": "Lord Krishna",
        "title_hi": "गोविंदा गोपाला | बाँसुरी और ढोल रस भजन",
        "title_en": "Govinda Gopala | Modern Melodic Krishna Bhajan",
        "short_caption": "राधे कृष्णा! बाल गोपाल और मुरली मनोहर का मधुर ऊर्जावान भजन 🙏 #Krishna #RadheKrishna #BhajanAabha",
        "bpm": 128,
        "prompt": (
            "Modern upbeat Hindi devotional Krishna bhajan, 128 BPM, 4/4, joyous uplifting male vocal, "
            "enchanting bansuri flute hooks, energetic groovy dholak and tabla, modern dance synths, "
            "euphoric festive chorus, rhythmic claps, clean and soulful Hindi singing. "
            "NOT mournful, NOT slow."
        ),
        "lyrics": """[Intro]
श्री कृष्ण गोविंद हरे मुरारी...
हे नाथ नारायण वासुदेवा...
राधे राधे... बोलो राधे राधे...

[Verse 1]
माखन चोर नंद किशोरा, मोर मुकुट बनवारी
तेरी मुरली की मधुर तान पे, झूमे दुनिया सारी
यमुना तीरे धेनु चरावे, गोपियों संग मुस्काए
जो भी ध्यावे मन से तुझको, भवसागर तर जाए

[Chorus]
गोविंदा गोपाला, मुरलीधर कान्हा
मन मंदिर में आजा, मेरे नटवर कान्हा
गोविंदा गोपाला, मुरलीधर कान्हा
राधे राधे बोलो, मेरे सांवरिया कान्हा

[Verse 2]
सुदामा की प्रीत निभाई, अर्जुन का रथ हांका
धर्म की रक्षा करने आए, रूप अनुपम झांकी
चरण कमल में नमन हमारा, हे करुणा के सागर
तेरी लगन में खो जाए ये, मन का सूना गागर

[Chorus]
गोविंदा गोपाला, मुरलीधर कान्हा
मन मंदिर में आजा, मेरे नटवर कान्हा
गोविंदा गोपाला, मुरलीधर कान्हा
राधे राधे बोलो, मेरे सांवरिया कान्हा

[Drop]
हरे कृष्णा हरे रामा... राधे राधे...

[Outro]
जय श्री कृष्णा... राधे राधे...""",
    },
    {
        "deity": "Lord Rama",
        "title_hi": "अयोध्या के राजा राम | मर्यादा पुरुषोत्तम भजन",
        "title_en": "Raghunandan Raja Ram | Royal Devotional DJ Bhajan",
        "short_caption": "श्री राम जय राम जय जय राम! मर्यादा पुरुषोत्तम राजा राम का पावन भजन 🙏 #JaiShreeRam #Ayodhya #BhajanAabha",
        "bpm": 128,
        "prompt": (
            "Modern high-energy Hindi devotional Rama bhajan, 128 BPM, 4/4, majestic and powerful male vocal, "
            "traditional dholak, celebratory nagada beats, temple bells, shehnai and synth brass layers, "
            "grand anthemic devotional chorus, pristine Hindi vocal diction and emotion. "
            "NOT low tempo, NOT silent."
        ),
        "lyrics": """[Intro]
श्री राम... श्री राम... जय जय राम...
मंगल भवन अमंगल हारी...

[Verse 1]
अयोध्या नगरी चमकी न्यारी, लौटे रघुकुल नंदन
चरण धूलि माथे पर धारे, करते प्रभु वंदन
धनुष बाण कर में सोहे, मुख पर दिव्य उजाला
भक्तों के संकट को काटे, कौशल्या का लाला

[Chorus]
श्री राम जय राम, जय जय राम
मेरे मन के दीप में, बसते श्री राम
श्री राम जय राम, जय जय राम
रघुनंदन राघव, राजा राम

[Verse 2]
मर्यादा की मूरत स्वामी, करुणा के अवतार
दीन दयाल कृपा सिंधु प्रभु, जग के पालनहार
तेरा नाम ही नौका प्रभु जी, तेरा नाम किनारा
राम नाम के सुमिरन से ही, संवरे जीवन सारा

[Chorus]
श्री राम जय राम, जय जय राम
मेरे मन के दीप में, बसते श्री राम
श्री राम जय राम, जय जय राम
रघुनंदन राघव, राजा राम

[Drop]
जय श्री राम... सियावर रामचंद्र की जय...

[Outro]
श्री राम जय राम जय जय राम... जय श्री राम...""",
    },
    {
        "deity": "Goddess Durga",
        "title_hi": "जय माँ भवानी | महिषासुर मर्दिनी शक्ति भजन",
        "title_en": "Maa Durga Bhavani | Powerful Electronic Devotional Bhajan",
        "short_caption": "जय माता दी! महिषासुर मर्दिनी माँ भवानी की शक्ति उपासना 🙏 #MaaDurga #Navratri #BhajanAabha",
        "bpm": 130,
        "prompt": (
            "Modern high-energy Hindi devotional Durga bhajan, 130 BPM, 4/4, fierce inspiring male devotional lead, "
            "pounding dhak and war drums, temple gongs, trishul resonant chiming, heavy EDM sub bass drops, "
            "soaring triumphant chorus, crisp Sanskrit-Hindi diction, dynamic festival atmosphere. "
            "NOT timid, NOT whispery."
        ),
        "lyrics": """[Intro]
या देवी सर्वभूतेषु शक्तिरूपेण संस्थिता...
नमस्तस्यै नमस्तस्यै नमस्तस्यै नमो नमः...
जय माता दी! जय माँ भवानी!

[Verse 1]
सिंह सवारी कर के आई, माँ जगदम्बे भवानी
अष्टभुजा में अस्त्र सजे हैं, महिमा अमर पुरानी
दुष्ट दलन संहारन मैया, भक्तों की रखवाली
तेरी कृपा से रोशन होती, जग की डाली डाली

[Chorus]
जय माँ अंबे, जय माँ दुर्गे
हे कल्याणी, जय माँ दुर्गे
जय माँ अंबे, जय माँ दुर्गे
सबकी झोली भरती मैया

[Verse 2]
महिषासुर को पल में मारा, देवों का मान बढ़ाया
जब भी विपदा आई जग पर, तेने हाथ उठाया
लाल चुनरिया शीश पे सोहे, नयनों में अमृत धारा
माँ की भक्ति में तर जाए, सारा जगत पसारा

[Chorus]
जय माँ अंबे, जय माँ दुर्गे
हे कल्याणी, जय माँ दुर्गे
जय माँ अंबे, जय माँ दुर्गे
सबकी झोली भरती मैया

[Drop]
जय दुर्गे... जय भवानी... जय माता दी!

[Outro]
बोलो सांचे दरबार की जय... जय माँ दुर्गे...""",
    },
    {
        "deity": "Goddess Kali",
        "title_hi": "महाकाली खप्पर वाली | रणचंडी शक्ति भजन",
        "title_en": "Mahakali Kaalratri | Thundering Devotional Bass Bhajan",
        "short_caption": "जय माँ महाकाली! दुष्ट विनाशिनी माँ कालरात्रि का प्रचंड स्तुति भजन 🙏 #MahaKali #Kaalratri #BhajanAabha",
        "bpm": 132,
        "prompt": (
            "Modern intense Hindi devotional Kali bhajan, 132 BPM, 4/4, commanding reverberant male vocal, "
            "thunderous damru and war drums, deep hypnotic basslines, sharp crash cymbals, fast dholak tempo, "
            "electrifying energetic chorus, pristine authentic Hindi devotional pronunciation. "
            "NOT slow, NOT dull."
        ),
        "lyrics": """[Intro]
ॐ क्रीं कालिकायै नमः...
जय माँ काली... जय महाकाली...

[Verse 1]
खप्पर वाली खड्ग विधारिणी, विकराल रूप तुम्हारा
मुंडमाल गल सोहे मैया, तू ही सत्य सहारा
काल का भी काल तू माता, भय को भगाने वाली
भक्तों के संकट को हरने, रण में आई काली

[Chorus]
जय महाकाली, भद्रकाली
जय कल्याणी, खप्पर वाली
जय महाकाली, भद्रकाली
मात हमारी, संकटहारी

[Verse 2]
रक्तबीज का मान मिटाया, चंड मुंड संहारे
तेरी शक्ति का कोई न पावे, पार हे जग उजियारे
शिव के उर पर पांव धरंती, शांत हुई महामाया
तेरी शरण में जो भी आया, निर्भय जीवन पाया

[Chorus]
जय महाकाली, भद्रकाली
जय कल्याणी, खप्पर वाली
जय महाकाली, भद्रकाली
मात हमारी, संकटहारी

[Drop]
ॐ जयंती मंगला काली भद्रकाली कपालिनी...

[Outro]
जय माँ काली... जय माँ आदिशक्ति...""",
    },
]


def get_daily_bhajan(day_offset: int = 0) -> BhajanThematicEntry:
    """Return the deterministic deity bhajan entry based on the current calendar day."""
    now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=5.5)  # IST
    day_index = (now.toordinal() + day_offset) % len(CATALOG)
    return CATALOG[day_index]
