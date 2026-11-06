"""Separate, disclosed experiment: citation traces left behind by the organisers' masking. Not part of the main (content-only) entry.

Two match types: `forward` = a citation in the candidate's header also appears as a reporter reference in the query text; `citator` = the candidate's CITATOR INFO names the query's own AIR citation.
Matched candidates are moved to the front of an existing ranking, keeping their relative order.
"""
import re

SCR = re.compile(r'\b(\d+)\s+S\.?\s?C\.?\s?R\.?\s+(\d+)\b')
AIR = re.compile(r'A\.?\s?I\.?\s?R\.?\s+(\d{4})\s+S\.?\s?C\.?\s+(\d+)\b')


def header_block(text):
    m = re.search(r'CITATION:(.*?)(?:CITATOR INFO|ACT:|HEADNOTE)', text[:1500], re.S)
    return m[1] if m else ''


def own_citations(text):
    """Citations of a document, from its own header."""
    h = header_block(text)
    return {('SCR', a, b) for a, b in re.findall(r'SCR\s*\((\d+)\)\s*(\d+)', h)} | {('AIR', y, p) for y, p in re.findall(r'(\d{4})\s+AIR\s+(\d+)', h)}


def references(text):
    """Reporter references in the body of a document (the header is skipped so a document's own citation does not count)."""
    t = text[1500:]
    return {('SCR', a, b) for a, b in SCR.findall(t)} | {('AIR', y, p) for y, p in AIR.findall(t)}


def citator(text):
    m = re.search(r'CITATOR INFO\s*:(.*?)(?:ACT:|HEADNOTE)', text[:2500], re.S)
    return {('AIR', y, p) for y, p in re.findall(r'(\d{4})\s*SC\s*(\d+)', m[1])} if m else set()


def matches(q, c, use=('forward', 'citator')):
    """{query id: set of matched candidate ids}. q: {id: dict(text=...)}, c: {id: text}."""
    own = {k: own_citations(v) for k, v in c.items()}; cit = {k: citator(v) for k, v in c.items()}; out = {}
    for k, v in q.items():
        ref, mine = references(v['text']), {x for x in own_citations(v['text']) if x[0] == 'AIR'}; hit = set()
        for x in c:
            if x == k: continue
            if ('forward' in use and own[x] & ref) or ('citator' in use and cit[x] & mine): hit.add(x)
        out[k] = hit
    return out


def promote(ranked, hits):
    """Matched candidates first (in their existing order, then any outside the list), then the rest."""
    out = {}
    for q, r in ranked.items():
        h = hits.get(q, set()); out[q] = [x for x in r if x in h] + sorted(h - set(r)) + [x for x in r if x not in h]
    return out


HEADER = re.compile(r'CITATION:.*?(?=ACT:|HEADNOTE|BENCH:|$)', re.S)
REPORTER = re.compile(r'\[?\(?\b\d{4}\b\)?\]?\s*\d*\s*(?:[A-Z]\.?\s?){2,6}\s*\(?[A-Za-z]*\)?\s*\d+|\b\d+\s+(?:[A-Z]\.\s?){2,6}\s*\d+\b|\b\d+\s+(?:[A-Z]{2,6}|[A-Z][a-z]+\.?\s?[A-Z]\.?)\s+\d+\b|A\.?\s?I\.?\s?R\.?\s+\d{4}\s+\w+\.?\s*\w*\.?\s+\d+')


def scrub(text):
    """Remove the header citation block (CITATION and CITATOR INFO) and reporter references from a document."""
    head, body = text[:2500], text[2500:]
    return REPORTER.sub(' ', HEADER.sub(' ', head, count=1) + body)
