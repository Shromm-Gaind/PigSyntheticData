// Copyright Epic Games, Inc. All Rights Reserved.

#include "CameraPoseCalcCharacter.h"
#include "CameraPoseCalcProjectile.h"
#include "Animation/AnimInstance.h"
#include "Camera/CameraComponent.h"
#include "Components/CapsuleComponent.h"
#include "EnhancedInputComponent.h"
#include "EnhancedInputSubsystems.h"
#include "GameFramework/Character.h"
#include "Camera/CameraComponent.h"
#include "MovieSceneSequencePlayer.h"




//////////////////////////////////////////////////////////////////////////
// ACameraPoseCalcCharacter

ACameraPoseCalcCharacter::ACameraPoseCalcCharacter()
{
	// Character doesnt have a rifle at start
	bHasRifle = false;
	
	// Set size for collision capsule
	GetCapsuleComponent()->InitCapsuleSize(55.f, 96.0f);
		
	// Create a CameraComponent	
	FirstPersonCameraComponent = CreateDefaultSubobject<UCameraComponent>(TEXT("FirstPersonCamera"));
	FirstPersonCameraComponent->SetupAttachment(GetCapsuleComponent());
	FirstPersonCameraComponent->SetRelativeLocation(FVector(-10.f, 0.f, 60.f)); // Position the camera
	FirstPersonCameraComponent->bUsePawnControlRotation = true;

	// Create a mesh component that will be used when being viewed from a '1st person' view (when controlling this pawn)
	Mesh1P = CreateDefaultSubobject<USkeletalMeshComponent>(TEXT("CharacterMesh1P"));
	Mesh1P->SetOnlyOwnerSee(true);
	Mesh1P->SetupAttachment(FirstPersonCameraComponent);
	Mesh1P->bCastDynamicShadow = false;
	Mesh1P->CastShadow = false;
	//Mesh1P->SetRelativeRotation(FRotator(0.9f, -19.19f, 5.2f));
	Mesh1P->SetRelativeLocation(FVector(-30.f, 0.f, -150.f));

}

void ACameraPoseCalcCharacter::BeginPlay()
{
	// Call the base class  
	Super::BeginPlay();

	//Add Input Mapping Context
	if (APlayerController* PlayerController = Cast<APlayerController>(Controller))
	{
		if (UEnhancedInputLocalPlayerSubsystem* Subsystem = ULocalPlayer::GetSubsystem<UEnhancedInputLocalPlayerSubsystem>(PlayerController->GetLocalPlayer()))
		{
			Subsystem->AddMappingContext(DefaultMappingContext, 0);
		}
	}

}

//////////////////////////////////////////////////////////////////////////// Input

void ACameraPoseCalcCharacter::SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent)
{
	// Set up action bindings
	if (UEnhancedInputComponent* EnhancedInputComponent = CastChecked<UEnhancedInputComponent>(PlayerInputComponent))
	{
		//Jumping
		EnhancedInputComponent->BindAction(JumpAction, ETriggerEvent::Triggered, this, &ACharacter::Jump);
		EnhancedInputComponent->BindAction(JumpAction, ETriggerEvent::Completed, this, &ACharacter::StopJumping);

		//Moving
		EnhancedInputComponent->BindAction(MoveAction, ETriggerEvent::Triggered, this, &ACameraPoseCalcCharacter::Move);

		//Looking
		EnhancedInputComponent->BindAction(LookAction, ETriggerEvent::Triggered, this, &ACameraPoseCalcCharacter::Look);
	}
}


void ACameraPoseCalcCharacter::Move(const FInputActionValue& Value)
{
	// input is a Vector2D
	FVector2D MovementVector = Value.Get<FVector2D>();

	if (Controller != nullptr)
	{
		// add movement 
		AddMovementInput(GetActorForwardVector(), MovementVector.Y);
		AddMovementInput(GetActorRightVector(), MovementVector.X);
	}
}

void ACameraPoseCalcCharacter::Look(const FInputActionValue& Value)
{
	// input is a Vector2D
	FVector2D LookAxisVector = Value.Get<FVector2D>();

	if (Controller != nullptr)
	{
		// add yaw and pitch input to controller
		AddControllerYawInput(LookAxisVector.X);
		AddControllerPitchInput(LookAxisVector.Y);
	}
}

void ACameraPoseCalcCharacter::SetHasRifle(bool bNewHasRifle)
{
	bHasRifle = bNewHasRifle;
}

bool ACameraPoseCalcCharacter::GetHasRifle()
{
	return bHasRifle;
}
/*
void ACameraPoseCalcCharacter::LogCharacterTransform()
{

	// Get the character's location and rotation
	FVector Location = GetActorLocation();
	FRotator Rotation = GetControlRotation();

	// Convert the character's location and rotation to strings
	FString LocationString = Location.ToString();
	FString RotationString = Rotation.ToString();

	// Log the character's location and rotation
	UE_LOG(LogTemp, Log, TEXT("Character Location: %s, Rotation: %s"), *LocationString, *RotationString);

	// Alternatively, you can concatenate the strings and print them in a single line
	FString PoseString = FString::Printf(TEXT("Character Location: %s, Rotation: %s"), *LocationString, *RotationString);
	UE_LOG(LogTemp, Log, TEXT("%s"), *PoseString);
}
void ACameraPoseCalcCharacter::LogCharacterTransform()
{
	// Get the First Person Camera component
	UCameraComponent* FirstPersonCamera = Cast<UCameraComponent>(GetComponentByClass(UCameraComponent::StaticClass()));
	if (FirstPersonCamera)
	{
		// Get the camera's location and rotation
		FVector Location = FirstPersonCamera->GetComponentLocation();
		FRotator Rotation = FirstPersonCamera->GetComponentRotation();

		// Convert the camera's location and rotation to strings
		FString LocationString = Location.ToString();
		FString RotationString = Rotation.ToString();

		// Format the log message
		FString LogMessage = FString::Printf(TEXT("Camera Location: %s, Rotation: %s\n"), *LocationString, *RotationString);

		// Write the log message to a text file
		FString LogFilePath = FPaths::ProjectSavedDir() / TEXT("CameraTransformLog.txt");
		FFileHelper::SaveStringToFile(LogMessage, *LogFilePath, FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), EFileWrite::FILEWRITE_Append);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("First Person Camera not found!"));
	}
}

void ACameraPoseCalcCharacter::StartLoggingCharacterTransform(float Interval)
{
	// Set a timer to call LogCharacterTransform periodically
	GetWorld()->GetTimerManager().SetTimer(TimerHandle, this, &ACameraPoseCalcCharacter::LogCharacterTransform, Interval, true);
}

*/



