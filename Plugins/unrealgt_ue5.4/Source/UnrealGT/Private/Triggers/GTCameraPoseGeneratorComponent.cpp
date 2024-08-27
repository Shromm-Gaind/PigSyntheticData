#include "Triggers/GTCameraPoseGeneratorComponent.h"

#include "Engine/World.h"

#include "GTFileUtilities.h"

UGTCameraPoseGeneratorComponent::UGTCameraPoseGeneratorComponent()
    : TargetLocation(FVector::ZeroVector)    
    , TargetRotation(FRotator::ZeroRotator)
{
    PrimaryComponentTick.bCanEverTick = false;
}
void UGTCameraPoseGeneratorComponent::GenerateData(const FDateTime& TimeStamp)
{
    AActor* Owner = GetOwner();
    if (Owner)
    {
        // Camera's current transform
        const FTransform& CameraTransform = Owner->GetActorTransform();

        // Camera's absolute position
        const FVector& CameraLocation = CameraTransform.GetLocation();
        const FQuat& CameraRotation = CameraTransform.GetRotation();

        // Convert quaternion to rotation matrix
        const FMatrix RotationMatrix = FRotationMatrix::Make(CameraRotation);

        // Camera direction, up, and right vectors in camera coordinate system
        const FVector CamDir = RotationMatrix.GetUnitAxis(EAxis::X);
        const FVector CamUp = RotationMatrix.GetUnitAxis(EAxis::Z);
        const FVector CamRight = RotationMatrix.GetUnitAxis(EAxis::Y);

        // Fixed vectors in world coordinate system
        const FVector CamLookAt(0, 0, 1);
        const FVector CamSky(0, 1, 0);

        // Fixed focal point in world coordinate system
        const FVector CamFPoint(0, 0, 10);

        // Prepare permutation matrix (identity matrix)
        const FMatrix IdentityMatrix = FMatrix::Identity;

        // Transform the permutation matrix to the camera coordinate system
        const FMatrix CameraCoordinatePermutationMatrix = IdentityMatrix * RotationMatrix;

        FString LogMessage;
        LogMessage.Reserve(1024);  // Reserve a reasonable size to avoid frequent reallocations

        // Check if target location and rotation are not null
        if (!TargetLocation.IsZero() || !TargetRotation.IsZero())
        {
            // Target transform
            const FTransform TargetTransform(TargetRotation, TargetLocation);

            // Calculate the relative transform of the camera to the target
            const FTransform ComputedRelativeTransform =
                CameraTransform.GetRelativeTransform(TargetTransform);

            // Extracting location and rotation from the computed relative transform
            const FVector& ComputedRelativeLocation = ComputedRelativeTransform.GetLocation();
            const FQuat& ComputedRelativeRotation = ComputedRelativeTransform.GetRotation();

            // Log relative location and rotation to the target
            LogMessage += FString::Printf(
                TEXT("Camera Relative Location to Target: %s, Camera Relative Rotation to Target: "
                     "%s\n"),
                *ComputedRelativeLocation.ToString(),
                *ComputedRelativeRotation.ToString());
        }

        // Log camera's absolute location, rotation matrix, and permutation matrix
        LogMessage += FString::Printf(
            TEXT("Camera Absolute Location: %s, Camera Absolute Rotation: %s, Timestamp: %s\n"),
            *CameraLocation.ToString(),
            *CameraRotation.ToString(),
            *TimeStamp.ToString());
        LogMessage += FString::Printf(
            TEXT("Camera Rotation Matrix:\n%s\n"), *MatrixToFormattedString(RotationMatrix));
        LogMessage += FString::Printf(
            TEXT("Camera Permutation Matrix (Camera Coordinate System):\n%s\n"),
            *PermutationMatrixToFormattedString(CameraCoordinatePermutationMatrix));

        // Log camera vectors
        LogMessage += FString::Printf(
            TEXT("Cam Pos: %s\nCam Dir: %s\nCam Up: %s\nCam Right: %s\n"),
            *CameraLocation.ToString(),
            *CamDir.ToString(),
            *CamUp.ToString(),
            *CamRight.ToString());
        LogMessage += FString::Printf(
            TEXT("Cam LookAt: %s\nCam Sky: %s\nCam FPoint: %s\nCam Angle: 90\n"),
            *CamLookAt.ToString(),
            *CamSky.ToString(),
            *CamFPoint.ToString());

        // Convert log message to byte array
        const TArray<uint8> Data = FGTFileUtilities::StringToCharArray(LogMessage);

        // Broadcast data ready event
        DataReadyDelegate.Broadcast(Data, TimeStamp);
    }
}

FString UGTCameraPoseGeneratorComponent::Vector3DToFormattedString(const FVector& InVector)
{
    return FString::Printf(TEXT("%f %f %f"), InVector.X, InVector.Y, InVector.Z);
}

FString UGTCameraPoseGeneratorComponent::RotatorToFormattedString(const FRotator& InRotator)
{
    return FString::Printf(TEXT("%f %f %f"), InRotator.Yaw, InRotator.Pitch, InRotator.Roll);
}


FString UGTCameraPoseGeneratorComponent::MatrixToFormattedString(const FMatrix& InMatrix)
{
    FString Result;
    Result.Reserve(128);  // Reserve a reasonable size to avoid frequent reallocations

    for (int32 Row = 0; Row < 3; ++Row)  // Only need 3x3 part of the 4x4 matrix
    {
        for (int32 Col = 0; Col < 3; ++Col)
        {
            Result += FString::Printf(TEXT("%f "), InMatrix.M[Row][Col]);
        }
        Result += TEXT("\n");
    }
    return Result;
}

FString UGTCameraPoseGeneratorComponent::PermutationMatrixToFormattedString(const FMatrix& InMatrix)
{
    FString Result;
    Result.Reserve(128);  // Reserve a reasonable size to avoid frequent reallocations

    for (int32 Row = 0; Row < 4; ++Row)
    {
        for (int32 Col = 0; Col < 4; ++Col)
        {
            Result += FString::Printf(TEXT("%f "), InMatrix.M[Row][Col]);
        }
        Result += TEXT("\n");
    }
    return Result;
}

