#include "Triggers/GTCameraPoseGeneratorComponent.h"

#include "Misc/FileHelper.h"
#include "HAL/PlatformFilemanager.h"

UGTCameraPoseGeneratorComponent::UGTCameraPoseGeneratorComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
}

void UGTCameraPoseGeneratorComponent::GenerateData(const FDateTime& TimeStamp)
{
    AActor* Owner = GetOwner();
    if (Owner)
    {
        // Hardcoded location and rotation values for the target
        FVector TargetLocation(-50, 1150, 60);
        FRotator TargetRotation(0, 0, 270);  // Assuming Pitch, Yaw, Roll format
        FTransform TargetTransform(TargetRotation, TargetLocation);

        // Camera's current transform
        FTransform CameraTransform = Owner->GetActorTransform();

        // Calculate the relative transform of the camera to the target
        FTransform ComputedRelativeTransform =
            CameraTransform.GetRelativeTransform(TargetTransform);

        // Extracting location and rotation from the computed relative transform
        FVector ComputedRelativeLocation = ComputedRelativeTransform.GetLocation();
        FQuat ComputedRelativeRotation = ComputedRelativeTransform.GetRotation();

        // Logging camera's absolute location and rotation
        FString CameraLocationString = CameraTransform.GetLocation().ToString();
        FString CameraRotationString =
            CameraTransform.GetRotation().ToString();  // Quaternion format

        // Logging computed relative location and quaternion rotation to the hardcoded target
        FString RelativeLocationString = ComputedRelativeLocation.ToString();
        FString RelativeRotationString = ComputedRelativeRotation.ToString();  // Quaternion format

        UE_LOG(
            LogTemp,
            Log,
            TEXT("Camera Absolute Location: %s, Camera Absolute Rotation: %s, Timestamp: %s"),
            *CameraLocationString,
            *CameraRotationString,
            *TimeStamp.ToString());
        UE_LOG(
            LogTemp,
            Log,
            TEXT("Camera Relative Location to Target: %s, Camera Relative Rotation to Target: %s"),
            *RelativeLocationString,
            *RelativeRotationString);

        FString LogMessage;
        LogMessage = FString::Printf(
            TEXT("Camera Absolute Location: %s, Camera Absolute Rotation: %s, Timestamp: %s\n"),
            *CameraLocationString,
            *CameraRotationString,
            *TimeStamp.ToString());
        LogMessage += FString::Printf(
            TEXT("Camera Relative Location to Target: %s, Camera Relative Rotation to Target: %s\n"),
            *RelativeLocationString,
            *RelativeRotationString);

        // Specify the file path for your log file
        FString FilePath = FPaths::ProjectDir() + TEXT("Logs/CameraLog.txt");

        // Append the log message to the file
        FFileHelper::SaveStringToFile(
            LogMessage,
            *FilePath,
            FFileHelper::EEncodingOptions::AutoDetect,
            &IFileManager::Get(),
            FILEWRITE_Append);
    }
}



/*


#include "GameFramework/Actor.h"

#include "Triggers/GTCameraPoseGeneratorComponent.h"

UGTCameraPoseGeneratorComponent::UGTCameraPoseGeneratorComponent()
{
    // Set this component to be initialized when the game starts, and to be ticked every frame.
    PrimaryComponentTick.bCanEverTick = false;
}



*/
