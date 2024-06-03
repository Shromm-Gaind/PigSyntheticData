// Fill out your copyright notice in the Description page of Project Settings.


#include "ActorMoverComponent.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"
#include "Misc/FileHelper.h"


// Sets default values for this component's properties
UActorMoverComponent::UActorMoverComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = false;

	// ...
}


void UActorMoverComponent::BeginPlay()
{
    Super::BeginPlay();
    AActor* Owner = GetOwner();
    if (Owner)
    {
        UE_LOG(LogTemp, Log, TEXT("Component attached to actor: %s"), *Owner->GetName());
        if (!ParseJSONFile(JsonFilePath))
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to parse JSON file: %s"), *JsonFilePath);
        }
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Component is not attached to any actor"));
    }
}


bool UActorMoverComponent::ParseJSONFile(const FString& FilePath)
{
    FString JsonString;
    if (FFileHelper::LoadFileToString(JsonString, *FilePath))
    {
        UE_LOG(LogTemp, Log, TEXT("JSON String: %s"), *JsonString);
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to load file: %s"), *FilePath);
        return false;
    }

    TSharedPtr<FJsonObject> JsonObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonString);
    TArray<TSharedPtr<FJsonValue>> JsonArray;
    if (!FJsonSerializer::Deserialize(Reader, JsonArray)) // Deserialize directly into an array
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to deserialize JSON content from file: %s"), *FilePath);
        return false;
    }

    for (const TSharedPtr<FJsonValue>& Value : JsonArray)
    {
        TSharedPtr<FJsonObject> PoseObject = Value->AsObject();

        // Parse Location
        TSharedPtr<FJsonObject> LocationObject = PoseObject->GetObjectField("CameraLocation");
        FVector Location(LocationObject->GetNumberField("X"),
            LocationObject->GetNumberField("Y"),
            LocationObject->GetNumberField("Z"));

        // Parse Rotation
        TSharedPtr<FJsonObject> RotationObject = PoseObject->GetObjectField("CameraRotation");
        FRotator Rotation(RotationObject->GetNumberField("P"),
            RotationObject->GetNumberField("Y"),
            RotationObject->GetNumberField("R"));

        CameraPoses.Add(FTransform(Rotation, Location));
    }
    return true;
}


void UActorMoverComponent::MoveActorToPose(int32 PoseIndex)
{
    UE_LOG(LogTemp, Log, TEXT("MoveActorToPose called with index: %d"), PoseIndex);

    if (!CameraPoses.IsValidIndex(PoseIndex))
    {
        UE_LOG(LogTemp, Warning, TEXT("Invalid pose index: %d"), PoseIndex);
        return;
    }

    AActor* Owner = GetOwner();
    if (!Owner)
    {
        UE_LOG(LogTemp, Error, TEXT("Owner is null"));
        return;
    }

    UE_LOG(LogTemp, Log, TEXT("Moving to pose: Location: %s, Rotation: %s"),
        *CameraPoses[PoseIndex].GetLocation().ToString(), *CameraPoses[PoseIndex].GetRotation().ToString());

    Owner->SetActorLocation(CameraPoses[PoseIndex].GetLocation());
    Owner->SetActorRotation(CameraPoses[PoseIndex].GetRotation());
}

int32 UActorMoverComponent::GetPoseCount() const
{
    return CameraPoses.Num();
}