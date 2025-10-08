"""Test script to verify NN metadata is stored and retrievable from Weaviate."""

import asyncio
import weaviate
from weaviate.classes.init import Auth
import json


async def test_metadata_retrieval():
    """Query Weaviate to verify nn_metadata is stored in chunks."""

    # Connect to Weaviate
    client = weaviate.connect_to_local(
        host="localhost",
        port=8080
    )

    try:
        collection = client.collections.get("Features_narodne_novine_hr")

        # Get first 5 objects to inspect metadata
        print("=" * 80)
        print("TESTING NARODNE NOVINE METADATA RETRIEVAL FROM WEAVIATE")
        print("=" * 80)
        print()

        response = collection.query.fetch_objects(limit=5)

        print(f"Retrieved {len(response.objects)} objects from Weaviate\n")

        for i, obj in enumerate(response.objects, 1):
            print(f"{'=' * 80}")
            print(f"CHUNK #{i}")
            print(f"{'=' * 80}")
            print(f"UUID: {obj.uuid}")
            print(f"\nProperties:")

            # Print all properties
            for key, value in obj.properties.items():
                if key == "nn_metadata":
                    print(f"\n  ✅ nn_metadata FOUND:")
                    # Pretty print the metadata
                    if isinstance(value, str):
                        try:
                            metadata_dict = json.loads(value)
                            print(f"      {json.dumps(metadata_dict, indent=6, ensure_ascii=False)}")
                        except:
                            print(f"      {value}")
                    else:
                        print(f"      {json.dumps(value, indent=6, ensure_ascii=False)}")
                elif key == "content":
                    print(f"  {key}: {str(value)[:100]}...")
                else:
                    print(f"  {key}: {value}")

            print()

        # Query specifically for documents with nn_metadata
        print(f"\n{'=' * 80}")
        print("CHECKING FOR nn_metadata FIELD IN ALL DOCUMENTS")
        print(f"{'=' * 80}\n")

        all_objects = collection.query.fetch_objects(limit=161)

        with_metadata = 0
        without_metadata = 0

        for obj in all_objects.objects:
            if "nn_metadata" in obj.properties and obj.properties["nn_metadata"]:
                with_metadata += 1
            else:
                without_metadata += 1

        print(f"Total chunks: {len(all_objects.objects)}")
        print(f"✅ Chunks WITH nn_metadata: {with_metadata}")
        print(f"❌ Chunks WITHOUT nn_metadata: {without_metadata}")

        if with_metadata > 0:
            print(f"\n🎉 SUCCESS: NN metadata is being stored in Weaviate!")
            print(f"Coverage: {with_metadata}/{len(all_objects.objects)} chunks ({100*with_metadata/len(all_objects.objects):.1f}%)")
        else:
            print(f"\n❌ FAILURE: No nn_metadata found in any chunks!")

    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(test_metadata_retrieval())
