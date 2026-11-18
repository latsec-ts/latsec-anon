"""vectors.py -- the LaTSEC test-vector generator.

Runs key generation and one signing session per profile with deterministic
randomness, and records the canonical encodings (or their digests).  The Rust
implementation (latsec-rs, `latsec kat`) writes the same files; both must
agree byte for byte.

Randomness: one SeededRandom stream (xof.py) with label "test-vector" and the
seed text, consumed in this order:

  1. Keygen: the 32-byte seed of A'; s (ell polynomials, D_{sigma_s});
     R ((k - 1) x ell polynomials, D_{sigma_R}).
  2. Signer sets of size t_sig (random_subset) until one is admissible.
  3. The 32-byte session identifier.
  4. Round 1 of each signer, in the order of act: r_i (ell polynomials,
     D_{sigma_r}).

The payload is "test vector for <profile id>".  Digests are SHAKE256 with
domain "LaTSEC-v1/test-vector" over the framed encodings, 32 bytes.

Usage:
  python3 vectors.py PROFILE.json [--seed TEXT]       print the vector
  python3 vectors.py --out DIR PROFILE.json ...       write DIR/<id>.json
  python3 vectors.py --verify VECTOR.json ...         regenerate and compare
"""

import argparse
import json
import os
import sys
import time

import latsec
from profiles import Profile
from xof import SeededRandom, Xof

FORMAT = "latsec-test-vector-v2"
KAT_DOMAIN = "LaTSEC-v1/test-vector"
INLINE_LIMIT = 1024
DEFAULT_SEED = "latsec-kat-1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def digest(parts):
    x = Xof(KAT_DOMAIN)
    for part in parts:
        x.absorb(part)
    return x.digest(32).hex()


def generate(profile, seed=DEFAULT_SEED):
    """One test vector, as an ordered dict."""
    scheme = latsec.Scheme(profile)
    rng = SeededRandom(seed.encode(), "test-vector")
    vk, abort_keys, shares = latsec.keygen(scheme, rng)
    signers = [latsec.Signer(scheme, vk, i, s_i) for i, s_i in enumerate(shares)]
    act = latsec.first_admissible_set(scheme, rng, profile.t_sig)
    sid = rng.read(32)
    payload = f"test vector for {profile.id}"

    msg, transcript, (kind, value) = latsec.sign_session(
        scheme, vk, abort_keys, signers, sid, act, payload.encode(), rng)
    if kind != "signature":
        raise RuntimeError(f"honest session aborted, blaming {value}")
    sig = value
    if not latsec.verify(scheme, vk, msg, sig):
        raise RuntimeError("signature does not verify")

    vk_bytes = vk.encode(scheme)
    ak_bytes = [latsec.encode_abort_key(scheme, i, t_i) for i, t_i in enumerate(abort_keys)]
    share_bytes = [latsec.encode_share(scheme, i, s_i) for i, s_i in enumerate(shares)]
    cmt_bytes = [cmt for cmt, _, _ in transcript]
    open_bytes = [latsec.encode_opening(scheme, w) for _, w, _ in transcript]
    resp_bytes = [latsec.encode_response(scheme, z) for _, _, z in transcript]
    sig_bytes = latsec.encode_signature(scheme, sig)
    _, _, lam, _ = latsec.decode_session_message(scheme, msg)

    def inline(data):
        return data.hex() if len(data) <= INLINE_LIMIT else None

    return {
        "format": FORMAT,
        "profile": profile.id,
        "profile_digest": profile.digest.hex(),
        "seed": seed,
        "signers": list(act),
        "session_id": sid.hex(),
        "payload": payload,
        "lambda": lam,
        "msg": msg.hex(),
        "bytes_msg": len(msg),
        "bytes_vk": len(vk_bytes),
        "bytes_abort_key": len(ak_bytes[0]),
        "bytes_share": len(share_bytes[0]),
        "bytes_commitment": len(cmt_bytes[0]),
        "bytes_opening": len(open_bytes[0]),
        "bytes_partial_response": len(resp_bytes[0]),
        "bytes_signature": len(sig_bytes),
        "digest_vk": digest([vk_bytes]),
        "digest_abort_keys": digest(ak_bytes),
        "digest_shares": digest(share_bytes),
        "digest_commitments": digest(cmt_bytes),
        "digest_openings": digest(open_bytes),
        "digest_partial_responses": digest(resp_bytes),
        "digest_signature": digest([sig_bytes]),
        "vk": inline(vk_bytes),
        "signature": inline(sig_bytes),
    }


def to_json(vector):
    """The same layout as the Rust writer: two-space indentation, with the
    integer lists `signers` and `lambda` on one line."""
    lines = ["{"]
    items = list(vector.items())
    for n, (key, value) in enumerate(items):
        if isinstance(value, list):
            text = "[" + ", ".join(str(x) for x in value) + "]"
        else:
            text = json.dumps(value)
        lines.append(f'  "{key}": {text}' + ("," if n + 1 < len(items) else ""))
    lines.append("}")
    return "\n".join(lines) + "\n"


def profile_for(vector_path, data):
    return Profile.load(os.path.join(REPO, "profiles", data["profile"] + ".json"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", help="profiles, or test vectors with --verify")
    ap.add_argument("--seed", default=DEFAULT_SEED)
    ap.add_argument("--out", help="directory to write <id>.json into")
    ap.add_argument("--verify", action="store_true", help="regenerate stored vectors and compare")
    args = ap.parse_args()
    failures = 0
    for path in args.files:
        started = time.time()
        if args.verify:
            with open(path) as f:
                text = f.read()
            stored = json.loads(text)
            fresh = generate(profile_for(path, stored), stored["seed"])
            ok = to_json(fresh) == text
            failures += not ok
            diff = [k for k in fresh if fresh[k] != stored.get(k)]
            print(f"{'OK  ' if ok else 'FAIL'} {path} ({time.time() - started:.1f} s)"
                  + ("" if ok else f" differs in {diff}"))
        else:
            vector = generate(Profile.load(path), args.seed)
            if args.out:
                target = os.path.join(args.out, vector["profile"] + ".json")
                with open(target, "w") as f:
                    f.write(to_json(vector))
                print(f"wrote {target} ({time.time() - started:.1f} s)")
            else:
                sys.stdout.write(to_json(vector))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
