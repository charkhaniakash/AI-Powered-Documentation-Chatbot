from app.services.duplicate_detector import get_duplicate_detector

def test_duplicate_detection():
    detector = get_duplicate_detector()
    
    # Test file hash
    file_hash1 = detector.calculate_file_hash("test.pdf")
    file_hash2 = detector.calculate_file_hash("test.pdf")
    assert file_hash1 == file_hash2
    print("✓ File hash consistent")
    
    # Test content hash
    content1 = "This is a test document"
    content2 = "This   is a   test   document"  # Extra spaces
    hash1 = detector.calculate_content_hash(content1)
    hash2 = detector.calculate_content_hash(content2)
    assert hash1 == hash2  # Normalized!
    print("✓ Content hash normalization works")
    
    # Test chunk deduplication
    from app.models.schemas import TextChunk, DocumentMetadata
    chunks = [
        TextChunk(chunk_id="c1", text="Same text", metadata=None),
        TextChunk(chunk_id="c2", text="Same text", metadata=None),
        TextChunk(chunk_id="c3", text="Different", metadata=None),
    ]
    unique = detector.deduplicate_chunks(chunks)
    assert len(unique) == 2  # Only 2 unique
    print("✓ Chunk deduplication works")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_duplicate_detection()