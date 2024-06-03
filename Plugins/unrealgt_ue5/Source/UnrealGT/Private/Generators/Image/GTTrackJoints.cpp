
#include "Generators/Image/GTTrackJoints.h"

#include "DrawDebugHelpers.h"
#include "EngineUtils.h"
#include "Generators/Image/GTSceneCaptureComponent2D.h"
#include "GameFramework/Actor.h"
#include "GTImage.h"
UGTTrackJoints::UGTTrackJoints()
{
    PrimaryComponentTick.bCanEverTick = true;
}

void UGTTrackJoints::BeginPlay()
{
    Super::BeginPlay();

    // HARDCODED SkeletalMeshComponent with 'SK_Pig'
    // This can be modified based on a dynamic search but I don't know how to do so
    // I have tried quite a few things, mostly it just wouldn't find the correct skeltal mesh


    for (TActorIterator<AActor> ActorItr(GetWorld()); ActorItr; ++ActorItr)
    {
        USkeletalMeshComponent* MeshComponent =
            ActorItr->FindComponentByClass<USkeletalMeshComponent>();
        if (MeshComponent && MeshComponent->SkeletalMesh &&
            MeshComponent->SkeletalMesh->GetName() == "SK_Pig")
        {
            FoundSkeletalMeshComponent = MeshComponent;  // Store the found component
            break;
        }
    }
}

void UGTTrackJoints::UpdateSocketLocations(USkeletalMeshComponent* SkeletalMeshComponent)
{
    if (!SkeletalMeshComponent)
    {
        UE_LOG(LogTemp, Error, TEXT("UGTTrackJoints: SkeletalMeshComponent is null."));
        return;
    }

    // Add a check for matching world contexts
    if (SkeletalMeshComponent->GetWorld() != GetWorld())
    {
        UE_LOG(LogTemp, Warning, TEXT("UGTTrackJoints: World context does not match."));
        return;
    }

    SocketLocations.Empty();
    for (const FName& SocketName : SkeletalMeshComponent->GetAllSocketNames())
    {
        FVector Location = SkeletalMeshComponent->GetSocketLocation(SocketName);
        SocketLocations.Add(Location);

        // Draw a debug point at the socket location
        //DrawDebugPoint(GetWorld(), Location, 10, FColor::Red, false, 30);
    }
}

void UGTTrackJoints::GenerateData(const FDateTime& TimeStamp)
{
    if (!FoundSkeletalMeshComponent)
    {
        UE_LOG(LogTemp, Warning, TEXT("UGTTrackJoints: No SkeletalMeshComponent found."));
        return;
    }
    UpdateSocketLocations(FoundSkeletalMeshComponent);

    if (!SceneCaptureComponent)
    {
        UE_LOG(LogTemp, Warning, TEXT("UGTTrackJoints: SceneCaptureComponent is null."));
        return;
    }

    // Project socket locations to pixel coordinates
    TArray<FVector2D> SocketPixelLocations;
    for (const FVector& Location : SocketLocations)
    {
        FVector2D PixelLocation;
        if (SceneCaptureComponent->ProjectToPixelLocation(Location, PixelLocation))
        {
            SocketPixelLocations.Add(PixelLocation);
        }
    }

    // Capture the image
    FGTImage CapturedImage;
    SceneCaptureComponent->CaptureImage(CapturedImage);
    TArray<FColor>& SegmentationData = CapturedImage.Pixels;
    int32 ImageWidth = CapturedImage.Width;
    int32 ImageHeight = CapturedImage.Height;

    // Modify regions around projected points in the captured image data
    int32 Radius = 3;  // Define the region size
    for (const FVector2D& PixelLocation : SocketPixelLocations)
    {
        FColor SocketColor = FColor::Red;  // Desired color or use ColorIndexCache[PrimColor]

        // Modify the pixel data in the region around each projected point
        for (int32 y = FMath::Max(0, PixelLocation.Y - Radius);
             y < FMath::Min(ImageHeight, PixelLocation.Y + Radius);
             ++y)
        {
            for (int32 x = FMath::Max(0, PixelLocation.X - Radius);
                 x < FMath::Min(ImageWidth, PixelLocation.X + Radius);
                 ++x)
            {
                int32 Index = y * ImageWidth + x;
                if (Index >= 0 && Index < SegmentationData.Num())
                {
                    SegmentationData[Index] = SocketColor;  // Modify the image data
                }
            }
        }
    }
    Super::GenerateData(TimeStamp);
    // Further processing with the captured image...
}


    // Any additional time-based logic can go here

