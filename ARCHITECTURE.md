# HindiBHajans — production architecture

## Daily production contract

The production workflow runs every day at **21:30 IST** and on manual dispatch.

Pipeline:

`approved singer image -> ACE-Step audio -> free Kaggle EchoMimicV3 GPU -> hard identity/singing QA -> master -> 3 Shorts -> release -> YouTube + Facebook + Instagram`

There is **no static-image video fallback**. A technically valid MP4 is not considered a valid production result unless the required person is visibly present and the visual QA gate passes.

## Production invariants

- ₹0 actual spend: no paid API, paid cloud compute, or automatic billing dependency.
- Only `assets/uks model image.png` is the singer identity source.
- The singer video must be audio-driven real facial/mouth motion, not a zoom/pan or slideshow substitute.
- The generated performance prompt requires one singer, traditional Indian clothing, and a Lord Ram devotional setting.
- No face-replacement fallback is permitted.
- The visual QA gate samples the final video, detects the reference identity with YuNet + SFace, requires the matched identity across the video, checks face scale, and requires visible lower-face motion.
- Shorts are derived only after the master passes QA, and each Short is independently QA-gated before publication.
- Any failed generation, unavailable free GPU, missing credential, identity mismatch, absent face, duplicate matched identity, or failed singing-motion check stops publication.
- Publishing steps are downstream of all hard QA gates.
- The master and Shorts are stored as GitHub Release assets so downstream platforms can fetch media without paid object storage.

## Zero-cost compute policy

The video worker is `app/kaggle_echomimic_public_dispatch.py`. It submits an open-source EchoMimicV3-Flash worker to Kaggle with a free NVIDIA GPU, downloads model weights directly from Hugging Face, generates short audio-driven segments, and concatenates them into the master. Kaggle's free GPU quota is finite; the architecture therefore **fails safely when quota or GPU availability is exhausted rather than substituting a lower-quality static video**.

The audio stage remains the ACE-Step hosted API configured by repository secrets. Only an explicitly configured no-charge endpoint is allowed.

## Safety and monetization posture

The architecture is designed to avoid the previous failure mode and to support original, non-repetitive production. It does not guarantee YPP approval. Photorealistic AI alteration/generation of a real person's appearance must be disclosed to YouTube where required, and the operator must have the necessary rights/permission for the person's likeness and all commercial audio/visual elements.
